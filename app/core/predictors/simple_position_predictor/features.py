import numpy as np
import pandas as pd
from app.config import DRIVERS_2025, PREDICTION_CONFIG

class FeatureHelper:
    def __init__(self, logger, feature_order: list[str] | None = None):
        self._log = logger.log
        self.feature_order = feature_order or []

    def active_weather(self) -> dict:
        scen_key = PREDICTION_CONFIG.get("active_scenario", "dry")
        return PREDICTION_CONFIG.get("weather_scenarios", {}).get(scen_key, {})

    def build_base_df(self) -> pd.DataFrame:
        race = PREDICTION_CONFIG["next_race"]
        w = self.active_weather()
        rows = []
        for code, cfg in DRIVERS_2025.items():
            rows.append({
                "driver": code,
                "team": cfg.get("team", ""),
                "rookie": cfg.get("rookie", False),
                "team_change": cfg.get("team_change", False),
                "race_name": race.get("race_name", ""),
                "year": race.get("year", 2025),
                "session_air_temp": w.get("session_air_temp", 25.0),
                "session_track_temp": w.get("session_track_temp", 35.0),
                "session_humidity": w.get("session_humidity", 60.0),
                "session_rainfall": float(w.get("session_rainfall", 0.0)) * 1.0,
            })
        df = pd.DataFrame(rows)
        df.index = df["driver"].values
        return df

    def align_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        order = self.feature_order or []
        if not order:
            return X
        X = X.copy()
        for col in order:
            if col not in X.columns:
                X[col] = 0.0
        return X[order]

    # --- weather perf (SIN shift) ---
    def add_weather_perf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        g = df.copy()
        for c in ["driver", "year", "points", "session_rainfall"]:
            if c not in g.columns:
                g[c] = np.nan
        g["points"] = pd.to_numeric(g["points"], errors="coerce").fillna(0.0)
        if g["session_rainfall"].dtype == bool:
            g["session_rainfall"] = g["session_rainfall"].astype(int)
        else:
            m = {"true":1, "1":1, "yes":1, "false":0, "0":0, "no":0}
            g["session_rainfall"] = g["session_rainfall"].apply(lambda x: m.get(str(x).strip().lower(), x))
            g["session_rainfall"] = pd.to_numeric(g["session_rainfall"], errors="coerce").fillna(0).astype(int)
        order_cols = ["year"]
        if "round" in g.columns and g["round"].notna().any():
            g["round"] = pd.to_numeric(g["round"], errors="coerce")
            order_cols.append("round")
        else:
            g["_row_ix_wthr"] = np.arange(len(g))
            order_cols.append("_row_ix_wthr")
        g = g.sort_values(order_cols).reset_index(drop=True)

        def _by_driver(h: pd.DataFrame) -> pd.DataFrame:
            h = h.copy()
            pts = h["points"].astype(float)
            rain = h["session_rainfall"].astype(int)
            dry = 1 - rain
            rain_pts_cumsum = (pts * rain).cumsum()
            rain_cnt_cumsum = rain.cumsum()
            dry_pts_cumsum = (pts * dry).cumsum()
            dry_cnt_cumsum = dry.cumsum()
            h["driver_avg_points_in_rain"] = np.where(rain_cnt_cumsum > 0, rain_pts_cumsum / rain_cnt_cumsum, 0.0)
            h["driver_avg_points_in_dry"] = np.where(dry_cnt_cumsum > 0, dry_pts_cumsum / dry_cnt_cumsum, 0.0)
            h["driver_rain_dry_delta"] = h["driver_avg_points_in_rain"] - h["driver_avg_points_in_dry"]
            return h

        g = g.groupby("driver", group_keys=False).apply(_by_driver)
        g.drop(columns=["_row_ix_wthr"], inplace=True, errors="ignore")
        return g
