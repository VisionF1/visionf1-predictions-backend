import numpy as np
from app.config import DATA_IMPORTANCE

from .hyperparameter_config import initialize_models_with_hyperparams
from .utils import get_cv_splitter, build_groups, calculate_year_weights
from .reporting import audit_features, show_detailed_comparison, show_model_results
from .persistence import save_model, save_metadata, save_training_results
from .model_train import train_single_model_with_cv


class ModelTrainer:
    def __init__(self, use_time_series_cv: bool = False):
        self.use_time_series_cv = use_time_series_cv
        self.models = initialize_models_with_hyperparams()
        self.results = {}

    def train_all_models(self, X_train, X_test, y_train, y_test, label_encoder, feature_names, df_original=None, train_indices=None):
        print("Entrenando modelos con Cross-Validation...")
        if X_train is None or y_train is None:
            print("❌ No hay datos para entrenar"); return {}

        # Pesos por año
        sample_weights = None
        if df_original is not None and 'year' in getattr(df_original, 'columns', []) and train_indices is not None:
            print("📅 Aplicando pesos por años a los datos de entrenamiento...")
            df_train = df_original.iloc[train_indices]
            sample_weights = calculate_year_weights(df_train)
        elif df_original is not None and 'year' in getattr(df_original, 'columns', []):
            print("📅 Aplicando pesos por años (sin índices, asumiendo orden)...")
            df_train = df_original.iloc[:len(X_train)]
            sample_weights = calculate_year_weights(df_train)
        else:
            print("⚠️ No se encontraron datos de años, usando pesos uniformes")

        if len(X_train) < 200:
            print("🛑 Dataset muy pequeño: Intente con mas datos")
            raise ValueError("Dataset muy pequeño para entrenar modelos de ML")
        elif len(X_train) < 500: # 24 Carreras * 20 pilotos = 480
            print("⚠️ Dataset pequeño: Es probable que haya overfitting")
        else:
            print("✅ Tamaño de dataset adecuado: ", len(X_train), "muestras")

        save_metadata(label_encoder, feature_names)
        try:
            target_name = getattr(y_train, 'name', None)
            audit_features(feature_names, target_name=target_name)
        except Exception as e:
            print(f"⚠️  Auditoría de features falló: {e}")

        # Groups y CV
        groups = build_groups(df_original, train_indices, X_train)
        cv_splitter = get_cv_splitter(X_train, groups=groups)

        # Entrenar cada modelo
        for name, model_config in self.models.items():
            print(f"\n{'='*50}\n🔍 ENTRENANDO {name} CON CROSS-VALIDATION\n{'='*50}")
            try:
                metrics, optimized_model = train_single_model_with_cv(
                    name, model_config, X_train, X_test, y_train, y_test, cv_splitter, sample_weights, groups
                )
                self.results[name] = metrics
                show_model_results(name, metrics)
                save_model(name, optimized_model)
            except Exception as e:
                print(f"❌ Error entrenando {name}: {e}")
                self.results[name] = { 'error': str(e) }

        show_detailed_comparison(self.results)
        save_training_results(self.results, label_encoder, feature_names)
        return self.results
