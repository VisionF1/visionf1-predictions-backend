import pandas as pd
from app.config import PREDICTION_CONFIG

class WeatherAdjuster:
    def __init__(self, logger):
        self._log = logger.log

    def apply(self, preds_df: pd.DataFrame, last_weather_stats: pd.DataFrame | None) -> pd.DataFrame:
        scen = PREDICTION_CONFIG.get("active_scenario", "dry")
        stats = last_weather_stats
        if stats is None or scen not in ("wet", "dry"):
            return preds_df
        alpha = 0  # sensibilidad (ajustable si quieres)
        sign  = -1.0 if scen == "wet" else +1.0
        m = stats.set_index("driver")["driver_rain_dry_delta"]
        out = preds_df.copy()
        out["driver_rain_dry_delta"] = out["driver"].map(m).fillna(0.0)
        out["predicted_position"] = (out["predicted_position"] + sign * alpha * (-out["driver_rain_dry_delta"])).astype(float)
        return out
