import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _format_lap_time(seconds: float) -> str:
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
class Fp3QualiModel:
    driver_ratio: Dict[str, float]
    driver_counts: Dict[str, int]
    team_ratio: Dict[str, float]
    global_ratio: float
    event_ratio: Dict[str, float]
    use_ratio: bool = True
    min_events: int = 2
    shrink_lambda: float = 0.3
    winsor_limits: tuple[float, float] = (0.05, 0.95)


class Fp3QualiPredictor:
    def __init__(self):
        self.model: Optional[Fp3QualiModel] = None

    @staticmethod
    def _winsorize(series: pd.Series, low_q: float, high_q: float) -> pd.Series:
        if series is None or series.empty:
            return series
        lo = series.quantile(low_q)
        hi = series.quantile(high_q)
        return series.clip(lower=lo, upper=hi)

    @staticmethod
    def _ensure_weekend_key(df: pd.DataFrame) -> pd.DataFrame:
        if "weekend_key" not in df.columns:
            wk = df.apply(lambda r: f"{int(r['year'])}_{str(r['race_name'])}", axis=1)
            df = df.copy()
            df["weekend_key"] = wk
        return df

    def fit(self,
            df: pd.DataFrame,
            use_ratio: bool = True,
            min_events: int = 2,
            shrink_lambda: float = 0.3,
            winsor_limits: tuple[float, float] = (0.05, 0.95)) -> None:
        """Entrena coeficientes robustos FP3→Quali por piloto/equipo/global.

        Espera columnas: driver, team, year, race_name, fp3_best_time, quali_best_time.
        Opcionales: weekend_key.
        """
        cur = df.copy()
        required = ["driver", "team", "year", "race_name", "fp3_best_time"]
        for col in required:
            if col not in cur.columns:
                raise ValueError(f"Falta columna requerida en dataset: {col}")

        # Derivar quali_best_time si es necesario
        if "quali_best_time" not in cur.columns:
            cand = []
            for c in ("q1_time", "q2_time", "q3_time", "quali_best_lap_from_laps"):
                if c in cur.columns:
                    cand.append(cur[c])
            if cand:
                cur["quali_best_time"] = pd.concat(cand, axis=1).min(axis=1)
            else:
                raise ValueError("No hay columnas de quali disponibles para derivar 'quali_best_time'")

        cur = self._ensure_weekend_key(cur)

        # Filtrar filas válidas
        cur["fp3_best_time"] = pd.to_numeric(cur["fp3_best_time"], errors="coerce")
        cur["quali_best_time"] = pd.to_numeric(cur["quali_best_time"], errors="coerce")
        cur = cur.dropna(subset=["fp3_best_time", "quali_best_time"]).copy()
        cur = cur[(cur["fp3_best_time"] > 0) & (cur["quali_best_time"] > 0)]
        if cur.empty:
            raise ValueError("Dataset vacío después de filtrar tiempos válidos")

        # Ratios y deltas
        cur["ratio_fp3_quali"] = cur["quali_best_time"] / cur["fp3_best_time"]
        cur["delta_fp3_quali_s"] = cur["quali_best_time"] - cur["fp3_best_time"]

        # Winsorizar
        rl, rh = winsor_limits
        cur["ratio_fp3_quali_w"] = self._winsorize(cur["ratio_fp3_quali"], rl, rh)
        cur["delta_fp3_quali_w_s"] = self._winsorize(cur["delta_fp3_quali_s"], rl, rh)

        # Estadísticas
        driver_counts = cur.groupby("driver")["ratio_fp3_quali_w"].size().to_dict()
        driver_ratio = cur.groupby("driver")["ratio_fp3_quali_w"].median().to_dict()
        team_ratio = cur.groupby("team")["ratio_fp3_quali_w"].median().to_dict()
        global_ratio = float(cur["ratio_fp3_quali_w"].median())
        event_ratio = cur.groupby("weekend_key")["ratio_fp3_quali_w"].median().to_dict()

        self.model = Fp3QualiModel(
            driver_ratio=driver_ratio,
            driver_counts=driver_counts,
            team_ratio=team_ratio,
            global_ratio=global_ratio,
            event_ratio=event_ratio,
            use_ratio=use_ratio,
            min_events=int(min_events),
            shrink_lambda=float(shrink_lambda),
            winsor_limits=winsor_limits,
        )

    def save(self, path: str = "app/models_cache/fp3_quali_model.pkl") -> None:
        if self.model is None:
            raise RuntimeError("Modelo no entrenado")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str = "app/models_cache/fp3_quali_model.pkl") -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return True

    def _ratio_for_row(self, row: pd.Series, weekend_key: str, scenario: str) -> float:
        m = self.model
        if m is None:
            raise RuntimeError("Modelo no cargado/entrenado")

        base = None
        d = str(row.get("driver", ""))
        t = str(row.get("team", ""))
        cnt = int(m.driver_counts.get(d, 0))
        if cnt >= m.min_events and d in m.driver_ratio:
            base = float(m.driver_ratio[d])
        elif t in m.team_ratio:
            base = float(m.team_ratio[t])
        else:
            base = float(m.global_ratio)

        # lambda ajustado por escenario
        lam = float(m.shrink_lambda)
        if scenario.lower() in ("wet", "storm"):
            lam = max(0.0, min(1.0, lam * 0.5))

        proxy = float(m.event_ratio.get(weekend_key, m.global_ratio))
        return (1.0 - lam) * base + lam * proxy

    def predict_event(self, fp3_df: pd.DataFrame, event_meta: Dict) -> pd.DataFrame:
        """Predice quali para un evento dado a partir de FP3 por piloto.

        fp3_df requiere: driver, team, year, race_name, fp3_best_time. Opcional: round, weekend_key
        event_meta: {year, race_name, round?, scenario?}
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado/entrenado")

        cur = fp3_df.copy()
        for col in ["driver", "team", "year", "race_name", "fp3_best_time"]:
            if col not in cur.columns:
                raise ValueError(f"Falta columna requerida en fp3_df: {col}")

        cur = self._ensure_weekend_key(cur)
        if "round" not in cur.columns and "round" in event_meta:
            cur["round"] = event_meta.get("round")

        scenario = str(event_meta.get("scenario", "dry"))
        cur["fp3_best_time"] = pd.to_numeric(cur["fp3_best_time"], errors="coerce")
        cur = cur.dropna(subset=["fp3_best_time"]).copy()
        cur = cur[cur["fp3_best_time"] > 0]
        if cur.empty:
            raise ValueError("Sin FP3 válidos para predecir")

        # ratio final y predicción
        cur["ratio_final"] = cur.apply(lambda r: self._ratio_for_row(r, r["weekend_key"], scenario), axis=1)
        cur["pred_best_quali_lap_s"] = cur["fp3_best_time"].astype(float) * cur["ratio_final"].astype(float)
        cur = cur.sort_values("pred_best_quali_lap_s", ascending=True).reset_index(drop=True)
        cur["pred_rank"] = np.arange(1, len(cur) + 1)
        cur["pred_best_quali_lap"] = cur["pred_best_quali_lap_s"].apply(_format_lap_time)

        out_cols = [
            "driver", "team", "race_name", "year",
            "round", "weekend_key", "pred_best_quali_lap_s", "pred_rank", "pred_best_quali_lap"
        ]
        for c in out_cols:
            if c not in cur.columns:
                cur[c] = None
        return cur[out_cols]

    def backtest(self, df_2025: pd.DataFrame) -> Dict[str, float]:
        """Backtest simple leave-one-event-out en 2025 devolviendo métricas globales."""
        cur = df_2025.copy()
        cur = self._ensure_weekend_key(cur)
        cur = cur.dropna(subset=["fp3_best_time", "quali_best_time"]).copy()
        cur = cur[(cur["fp3_best_time"] > 0) & (cur["quali_best_time"] > 0)]
        if cur.empty:
            return {"mae_s": np.nan, "medae_s": np.nan, "events": 0}

        events = sorted(cur["weekend_key"].unique())
        errors = []
        for wk in events:
            train = cur[cur["weekend_key"] != wk]
            test = cur[cur["weekend_key"] == wk]
            if len(train) < 5 or len(test) == 0:
                continue
            # Entrenar temporalmente
            tmp = Fp3QualiPredictor()
            tmp.fit(train,
                    use_ratio=self.model.use_ratio if self.model else True,
                    min_events=self.model.min_events if self.model else 2,
                    shrink_lambda=self.model.shrink_lambda if self.model else 0.3,
                    winsor_limits=self.model.winsor_limits if self.model else (0.05, 0.95))
            # Predicción
            fp3_ev = test[["driver", "team", "year", "race_name", "fp3_best_time"]].copy()
            meta = {
                "year": int(test["year"].iloc[0]),
                "race_name": str(test["race_name"].iloc[0]),
                "round": int(test.get("round", pd.Series([None])).iloc[0] or 0),
                "scenario": "dry",
            }
            preds = tmp.predict_event(fp3_ev, meta)
            merged = preds.merge(test[["driver", "quali_best_time"]], on="driver", how="inner")
            if not merged.empty:
                e = (merged["pred_best_quali_lap_s"].astype(float) - merged["quali_best_time"].astype(float)).abs().values
                errors.extend(e.tolist())

        if not errors:
            return {"mae_s": np.nan, "medae_s": np.nan, "events": 0}
        mae = float(np.mean(errors))
        medae = float(np.median(errors))
        return {"mae_s": mae, "medae_s": medae, "events": len(events)}

