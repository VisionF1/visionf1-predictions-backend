import os
import pickle
from .paths import MODEL_FILES, TRAINING_RESULTS_PKL

class ModelLoader:
    def __init__(self, logger):
        self._log = logger.log

    def load_model(self, name: str | None):
        candidates = []
        if name and name in MODEL_FILES:
            candidates.append((name, MODEL_FILES[name]))
        for n, p in MODEL_FILES.items():
            if not any(p == c[1] for c in candidates):
                candidates.append((n, p))
        for n, path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        model = pickle.load(f)
                    print(f"✅ Modelo cargado: {n}")
                    return model
                except Exception as e:
                    print(f"⚠️  Error cargando {n}: {e}")
        print("❌ No se encontró ningún modelo entrenado.")
        return None

    def _load_training_metrics(self, path: str = TRAINING_RESULTS_PKL):
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            self._log(f"⚠️  No se pudieron leer métricas: {e}")
        return None

    def _select_best_from_metrics(self, metrics_dict: dict) -> str | None:
        candidates = []
        for name, m in metrics_dict.items():
            if not isinstance(m, dict) or ("error" in m):
                continue
            cv = m.get("cv_mse_mean", float("inf"))
            over = m.get("overfitting_score", 1.0)
            r2 = m.get("test_r2", 0.0)
            comp = (cv / 100.0) + max(0.0, over - 1.0) * 2.0 + (1.0 - max(0.0, min(1.0, r2))) * 10.0
            candidates.append((comp, name))
        if not candidates:
            return None
        candidates.sort()
        best = candidates[0][1]
        print(f"🎯 Mejor modelo por métricas: {best}")
        return best

    def load_best_model(self):
        metrics = self._load_training_metrics()
        if not metrics:
            return None
        best_name = self._select_best_from_metrics(metrics)
        if not best_name:
            return None
        return self.load_model(best_name)
