import numpy as np
import pandas as pd
from app.config import PREDICTION_CONFIG, PENALTIES
from app.core.adapters.progressive_adapter import ProgressiveAdapter
from .paths import INFERENCE_OUT
from .manifest import ManifestHelper
from .models import ModelLoader
from .features import FeatureHelper
from .dataset import InferenceDatasetBuilder
from .weather_adjust import WeatherAdjuster
from .utils import deterministic_eps, Logger, Explainer

class SimplePositionPredictor:
    def __init__(self, quiet: bool = False):
        self.logger = Logger(quiet=quiet)
        self._log = self.logger.log
        self.mh = ManifestHelper(self.logger)

        forced_model_name = None
        if self.mh.manifest and isinstance(self.mh.manifest.get("best_model_name"), str):
            forced_model_name = self.mh.manifest["best_model_name"]
            self._log(f"🧭 Manifiesto sugiere modelo: {forced_model_name}")

        self.ml = ModelLoader(self.logger)
        if forced_model_name:
            self.model = self.ml.load_model(forced_model_name) or self.ml.load_best_model() or self.ml.load_model(None)
        else:
            self.model = self.ml.load_best_model() or self.ml.load_model(None)

        # orden de features: manifiesto > pkl legacy
        feature_order = self.mh.feature_names_from_manifest() or self.mh.load_trained_feature_names()
        self.fh = FeatureHelper(self.logger, feature_order)
        self.db = InferenceDatasetBuilder(self.logger, self.fh, self.mh)
        self.adapter = ProgressiveAdapter()
        self.weather_adj = WeatherAdjuster(self.logger)
        self.explainer = Explainer(self.logger, self.fh, self.db, self.model)

    # -------------------- API principal --------------------
    def predict_positions_2025(self) -> pd.DataFrame:
        print("🎯 Prediciendo posiciones (pre-race)")
        base_df = self.fh.build_base_df()
        X = self.db.build_X(base_df)

        # Export dataset con meta para inspección, igual que antes
        try:
            X_with_meta = X.copy()
            if "driver" not in X_with_meta.columns:
                X_with_meta.insert(0, "driver", X.index)
            if "team" not in X_with_meta.columns:
                X_with_meta.insert(1, "team", base_df.loc[X.index, "team"].values)
            stats = self.db.last_weather_stats
            if stats is not None and not stats.empty:
                X_with_meta = X_with_meta.merge(stats, on="driver", how="left")
            else:
                for c in ["driver_avg_points_in_rain", "driver_avg_points_in_dry", "driver_rain_dry_delta"]:
                    if c not in X_with_meta.columns:
                        X_with_meta[c] = 0.0
            X_with_meta.to_csv(INFERENCE_OUT, index=False)
        except Exception as e:
            pass
        
        if X is None or X.shape[0] == 0:
            self._log("⚠️ X vacío: uso predicción determinista como respaldo")
            drivers = base_df.index.tolist()
            y_hat = np.array([10.0 + deterministic_eps(d) for d in drivers])
            idx = drivers
        else:
            idx = X.index
            if self.model is None:
                y_hat = np.array([10.0 + deterministic_eps(d) for d in idx])
            else:
                y_hat = self.model.predict(X.values)

        out = pd.DataFrame({
            "driver": idx,
            "team": base_df.loc[idx, "team"].values,
            "rookie": base_df.loc[idx, "rookie"].values,
            "team_change": base_df.loc[idx, "team_change"].values,
            "predicted_position": y_hat.astype(float),
        })

        out["predicted_position"] = out.apply(
            lambda r: float(r["predicted_position"]) + deterministic_eps(r["driver"]), axis=1
        )

        # Ajuste post-modelo por clima
        try:
            out = self.weather_adj.apply(out, self.db.last_weather_stats)
        except Exception as e:
            self._log(f"ℹ️ Ajuste climático omitido: {e}")

        # Adaptaciones progresivas
        if PENALTIES.get("use_progressive", False):
            race_no = int(PREDICTION_CONFIG["next_race"].get("race_number", 1))
            out = self.adapter.apply_progressive_penalties(out, race_no)

        out = out.sort_values("predicted_position", ascending=True).reset_index(drop=True)
        out["final_position"] = np.arange(1, len(out) + 1)
        out["confidence"] = out.apply(
            lambda x: max(60, 100 - abs(x["predicted_position"] - x["final_position"]) * 10), axis=1
        ).round(1)

        def _type(row):
            if row.get("rookie", False):
                return "🆕 Rookie"
            if row.get("team_change", False):
                return "🔄 Cambio equipo"
            return "👤 Veterano"
        out["driver_type"] = out.apply(_type, axis=1)

        return out[["final_position", "driver", "team", "driver_type", "predicted_position", "confidence"]]

    # -------------------- compatibilidad --------------------
    def show_realistic_predictions(self, predictions_df: pd.DataFrame) -> None:
        current_race_name = PREDICTION_CONFIG["next_race"].get("race_name", "Carrera Desconocida")
        print(f"\n{'='*100}")
        print(f"🏆 PREDICCIONES 2025 - {current_race_name}")
        print(f"{'='*100}")
        print(f"{'Pos':<4} {'Piloto':<6} {'Equipo':<16} {'Tipo':<20} {'Pred':<8} {'Conf.':<6}")
        print("-" * 100)
        for _, row in predictions_df.iterrows():
            print(
                f"P{int(row['final_position']):<3} {row['driver']:<6} {str(row['team'])[:16]:<16} {row['driver_type']:<20} "
                f"{float(row['predicted_position']):<8.3f} {float(row['confidence']):<6.1f}"
            )

    # -------------------- explicabilidad --------------------
    def explain_feature_importance(self, *args, **kwargs):
        return self.explainer.explain_feature_importance(*args, **kwargs)
