import hashlib

def deterministic_eps(driver_code: str) -> float:
    h = hashlib.sha1(driver_code.encode("utf-8")).hexdigest()
    return (int(h[:6], 16) % 1000) * 1e-6


class Logger:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
    def log(self, msg: str):
        if not self.quiet:
            print(msg)



import os
import pandas as pd
from .paths import INFERENCE_OUT

class Explainer:
    def __init__(self, logger, feature_helper, dataset_builder, model):
        self._log = logger.log
        self.fh = feature_helper
        self.db = dataset_builder
        self.model = model

    def _compute_permutation_importance_on_preds(self, X: pd.DataFrame, n_repeats: int = 10, random_state: int = 42) -> pd.Series:
        import numpy as np
        rng = np.random.RandomState(random_state)
        X_aligned = self.fh.align_columns(X)
        cols = list(X_aligned.columns)
        baseline = self.model.predict(X_aligned.values).astype(float)
        changes = np.zeros(len(cols), dtype=float)
        for rep in range(n_repeats):
            for j, col in enumerate(cols):
                Xp = X_aligned.copy()
                Xp[col] = rng.permutation(Xp[col].values)
                preds = self.model.predict(Xp.values).astype(float)
                changes[j] += np.mean(np.abs(preds - baseline))
        changes /= float(n_repeats)
        return pd.Series(changes, index=cols)

    def explain_feature_importance(self, top_k: int = 25,
                                   csv_path: str = "app/models_cache/feature_importances.csv",
                                   png_path: str = "app/models_cache/feature_importances.png",
                                   n_repeats: int = 10) -> pd.DataFrame:
        import matplotlib.pyplot as plt
        base_df = self.fh.build_base_df()
        X = self.db.build_X(base_df)
        X = self.fh.align_columns(X)
        if hasattr(self.model, "feature_importances_") and self.model.feature_importances_ is not None:
            importances = pd.Series(self.model.feature_importances_, index=X.columns, dtype=float)
            method = "native"
        else:
            importances = self._compute_permutation_importance_on_preds(X, n_repeats=n_repeats)
            method = f"permutation_preds_{n_repeats}"
        fi = importances.sort_values(ascending=False).to_frame(name="importance")
        fi["importance_pct"] = (fi["importance"] / fi["importance"].sum() * 100.0).round(2)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        fi.reset_index(names="feature").to_csv(csv_path, index=False)
        try:
            top = fi.head(top_k)
            plt.figure()
            plt.barh(top.index[::-1], top["importance"].iloc[:top_k][::-1])
            plt.title(f"Feature Importance ({method}) — top {top_k}")
            plt.xlabel("importance")
            plt.ylabel("feature")
            plt.tight_layout()
            plt.savefig(png_path, dpi=140)
        except Exception as e:
            self._log(f"⚠️ No pude generar gráfico de importancias: {e}")
        self._log(f"💾 Feature importances guardadas: {csv_path}")
        self._log(f"🖼️ Gráfico guardado: {png_path}")
        return fi

