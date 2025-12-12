import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Iterable

import numpy as np
import pandas as pd

from app.config import DRIVERS_2025


def _fmt_lap(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return ""
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{m:02d}:{s:02d}:{ms:03d}"


@dataclass
class QualiRecentModel:
    driver_delta_median_s: Dict[str, float]
    driver_rank_median: Dict[str, float]
    team_delta_median_s: Dict[str, float]
    global_delta_median_s: float
    anchor_best_by_race: Dict[str, float]  # race_name -> median of best quali times
    n_recent: int = 3


class RecentQualiPredictor:
    def __init__(self):
        self.model: Optional[QualiRecentModel] = None

    @staticmethod
    def _ensure_weekend_key(df: pd.DataFrame) -> pd.DataFrame:
        if "weekend_key" not in df.columns:
            df = df.copy()
            df["weekend_key"] = df.apply(lambda r: f"{int(r['year'])}_{str(r['race_name'])}", axis=1)
        return df

    def fit(self, df: pd.DataFrame, n_recent: int = 3) -> None:
        cur = df.copy()
        required = ["driver", "team", "race_name", "year"]
        for c in required:
            if c not in cur.columns:
                raise ValueError(f"Falta columna requerida: {c}")
        # derive quali_best_time
        if "quali_best_time" not in cur.columns:
            cands = []
            for c in ("q1_time", "q2_time", "q3_time", "quali_best_lap_from_laps"):
                if c in cur.columns:
                    cands.append(pd.to_numeric(cur[c], errors="coerce"))
            if cands:
                cur["quali_best_time"] = pd.concat(cands, axis=1).min(axis=1)
        cur["quali_best_time"] = pd.to_numeric(cur.get("quali_best_time", np.nan), errors="coerce")

        # sprint quali best time (optional)
        cur["sq_best_time"] = pd.to_numeric(cur.get("sq_best_time", np.nan), errors="coerce")

        # reference time: prefer sq_best_time if present else quali_best_time
        cur["ref_best_time"] = cur["sq_best_time"].where(cur["sq_best_time"].notna(), cur["quali_best_time"])

        cur = cur.dropna(subset=["ref_best_time"]).copy()
        cur = cur[cur["ref_best_time"] > 0]
        if cur.empty:
            raise ValueError("Dataset vacío sin tiempos de quali")

        cur = self._ensure_weekend_key(cur)

        # round ordering
        if "round" in cur.columns:
            cur["_sort_round"] = pd.to_numeric(cur["round"], errors="coerce").fillna(0)
        else:
            cur["_sort_round"] = 0
        cur = cur.sort_values(["year", "_sort_round"]).reset_index(drop=True)

        # event best anchor per weekend
        best_by_wk = cur.groupby("weekend_key")["ref_best_time"].min().rename("event_best_time")
        cur = cur.merge(best_by_wk, on="weekend_key", how="left")
        cur["delta_to_event_best_s"] = cur["ref_best_time"] - cur["event_best_time"]

        # last n events per driver
        cur["_row_ix"] = np.arange(len(cur))
        def _last_n(h: pd.DataFrame) -> pd.DataFrame:
            return h.sort_values(["year", "_sort_round", "_row_ix"]).tail(n_recent)
        last_n = cur.groupby("driver", group_keys=False).apply(_last_n)

        driver_delta_median_s = last_n.groupby("driver")["delta_to_event_best_s"].median().to_dict()
        # If quali_position exists, compute rank median, else derive approximate rank by sorting by time per weekend
        if "quali_position" in last_n.columns:
            driver_rank_median = last_n.groupby("driver")["quali_position"].median().to_dict()
        else:
            # approximate rank by ranking times within each weekend
            tmp = last_n.copy()
            tmp["approx_rank"] = tmp.groupby("weekend_key")["quali_best_time"].rank(method="min")
            driver_rank_median = tmp.groupby("driver")["approx_rank"].median().to_dict()

        team_delta_median_s = last_n.groupby("team")["delta_to_event_best_s"].median().to_dict()
        global_delta_median_s = float(last_n["delta_to_event_best_s"].median())

        # anchor by race_name across all known years (ref_best_time)
        anchor_best_by_race = cur.groupby("race_name")["event_best_time"].median().to_dict()

        # incorporate sprint performance as minor adjustment signal
        sprint_bonus = {}
        if "sprint_position" in df.columns:
            tmp = df.copy()
            tmp["sprint_position"] = pd.to_numeric(tmp.get("sprint_position", np.nan), errors="coerce")
            tmp = tmp.dropna(subset=["sprint_position"]) if not tmp.empty else tmp
            if tmp is not None and not tmp.empty:
                sprint_bonus = tmp.groupby("driver")["sprint_position"].median().to_dict()

        self.model = QualiRecentModel(
            driver_delta_median_s=driver_delta_median_s,
            driver_rank_median=driver_rank_median,
            team_delta_median_s=team_delta_median_s,
            global_delta_median_s=global_delta_median_s,
            anchor_best_by_race=anchor_best_by_race,
            n_recent=int(n_recent),
        )
        # attach bonus map lazily
        self._sprint_bonus = sprint_bonus

    def save(self, path: str = "app/models_cache/quali_recent_model.pkl") -> None:
        if self.model is None:
            raise RuntimeError("Modelo no entrenado")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str = "app/models_cache/quali_recent_model.pkl") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return True

    def predict_next_event(self, event_meta: Dict, event_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Modelo no cargado/entrenado")
        m = self.model
        race_name = str(event_meta.get("race_name", ""))
        year = int(event_meta.get("year", 2025))
        round_num = event_meta.get("round")

        # anchor time (ref: prefer sprint-qualifying anchor if historically present; else quali anchor)
        anchor = float(m.anchor_best_by_race.get(race_name, np.nan))
        if not np.isfinite(anchor):
            # fallback: global median of event bests
            anchor = float(np.median(list(m.anchor_best_by_race.values()))) if m.anchor_best_by_race else 80.0

        # build driver table from DRIVERS_2025
        rows = []
        for code, cfg in DRIVERS_2025.items():
            d = code
            team = cfg.get("team", None)
            # choose delta: driver > team > global
            if d in m.driver_delta_median_s and np.isfinite(m.driver_delta_median_s[d]):
                delta = float(m.driver_delta_median_s[d])
            elif team in m.team_delta_median_s and np.isfinite(m.team_delta_median_s[team]):
                delta = float(m.team_delta_median_s[team])
            else:
                delta = float(m.global_delta_median_s)
            # small sprint performance adjustment (lower sprint pos => smaller delta)
            try:
                sp = self._sprint_bonus.get(d) if hasattr(self, "_sprint_bonus") else None
                if sp is not None and np.isfinite(sp):
                    # center around mid-grid 10.5; adjust by 0.01s per position
                    delta += (sp - 10.5) * 0.01
            except Exception:
                pass
            pred_time = max(0.0, anchor + delta)
            rows.append({
                "driver": d,
                "team": team,
                "race_name": race_name,
                "year": year,
                "round": round_num,
                "pred_best_quali_lap_s": pred_time,
            })

        out = pd.DataFrame(rows)
        # refine with current event sprint/sq info if available
        if event_df is not None and not event_df.empty:
            df_ev = event_df.copy()
            # Prefer direct sq_best_time when present
            if "sq_best_time" in df_ev.columns:
                try:
                    df_ev["sq_best_time"] = pd.to_numeric(df_ev["sq_best_time"], errors="coerce")
                    out = out.merge(df_ev[["driver", "sq_best_time"]], on="driver", how="left")
                    mask = out["sq_best_time"].notna() & (out["sq_best_time"] > 0)
                    out.loc[mask, "pred_best_quali_lap_s"] = out.loc[mask, "sq_best_time"].astype(float)
                    out = out.drop(columns=["sq_best_time"], errors="ignore")
                except Exception:
                    pass
            # Small tweak by sprint_position if present
            if "sprint_position" in df_ev.columns:
                try:
                    out = out.merge(df_ev[["driver", "sprint_position"]], on="driver", how="left")
                    out["sprint_position"] = pd.to_numeric(out["sprint_position"], errors="coerce")
                    out["pred_best_quali_lap_s"] = out["pred_best_quali_lap_s"].astype(float) + (out["sprint_position"] - 10.5).fillna(0) * 0.01
                    out = out.drop(columns=["sprint_position"], errors="ignore")
                except Exception:
                    pass
        out = out.sort_values("pred_best_quali_lap_s", ascending=True).reset_index(drop=True)
        out["pred_rank"] = np.arange(1, len(out) + 1)
        out["pred_best_quali_lap"] = out["pred_best_quali_lap_s"].apply(_fmt_lap)
        out["weekend_key"] = out.apply(lambda r: f"{int(r['year'])}_{str(r['race_name'])}", axis=1)
        return out[["driver", "team", "race_name", "year", "round", "weekend_key", "pred_best_quali_lap_s", "pred_rank", "pred_best_quali_lap"]]
