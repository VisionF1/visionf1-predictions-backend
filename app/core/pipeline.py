import pickle
import os
import pandas as pd
from datetime import datetime
from app.data.collectors.fastf1_collector import FastF1Collector
from app.data.preprocessors.data_cleaner import clean_data
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
from app.core.training.model_trainer import ModelTrainer
from app.core.utils.race_range_builder import RaceRangeBuilder
from app.core.predictors.simple_position_predictor.predictor import SimplePositionPredictor
from app.core.predictors.fp3_quali_predictor import Fp3QualiPredictor
from app.core.predictors.quali_recent_predictor import RecentQualiPredictor
from app.config import PREDICTION_CONFIG, RACE_PREDICTION

class Pipeline:
    """Pipeline principal con features avanzadas"""
    
    def __init__(self, config):
        self.config = config
        self.data = None
        
        # Componentes especializados con features avanzadas
        self.race_range_builder = RaceRangeBuilder()
        self.data_preparer = EnhancedDataPreparer()
        self.model_trainer = ModelTrainer()
        
        # Collector
        race_range = self.race_range_builder.build_race_range(config)
        self.collector = FastF1Collector(race_range)

    def run(self):
        
        """Ejecuta el pipeline completo con validación robusta"""
        print("🚀 Iniciando pipeline de entrenamiento...")
        
        # 1. Cargar o recolectar datos
        if not self._load_cached_data():
            print("📥 Recolectando datos frescos...")
            self.collect_data()
            self.preprocess_data()
            self._save_cached_data()

        
        
        # 2. Validar tamaño del dataset
        if self.data is None or len(self.data) < 30:
            data_len = len(self.data) if self.data is not None else 0
            print(f"⚠️  ADVERTENCIA: Dataset pequeño ({data_len} muestras)")
            print(f"   💡 Considera recolectar más datos para evitar overfitting")
        
        # 3. Guardar dataset original antes de entrenamiento
        self._save_dataset_before_training()
        
        # 4. Preparar datos para entrenamiento
        training_results = self.data_preparer.prepare_training_data(self.data)
        if training_results[0] is None:
            print("❌ Error preparando datos de entrenamiento")
            return False
        
        X_train, X_test, y_train, y_test, feature_names = training_results
        
        # 5. Guardar dataset después del feature engineering
        self._save_dataset_after_feature_engineering(X_train, X_test, y_train, y_test)

        # 6. Validar split de datos
        print(f"📊 Datos de entrenamiento: {len(X_train)} muestras")
        print(f"📊 Datos de test: {len(X_test)} muestras")
        
        if len(X_train) < 20:
            print(f"🚨 ADVERTENCIA: Muy pocos datos de entrenamiento")
            print(f"   💡 Cross-validation será limitado")
        
        # 7. Entrenar modelos con cross-validation (incluyendo pesos por años)
        model_trainer = ModelTrainer(use_time_series_cv=True)
        
        # Debug: verificar si los índices están disponibles
        train_indices = getattr(self.data_preparer, 'train_indices', None)
        print(f"🔍 DEBUG: train_indices en pipeline: {train_indices is not None}")
        if train_indices is not None:
            print(f"🔍 DEBUG: len(train_indices): {len(train_indices)}")
        
        results = model_trainer.train_all_models(
            X_train, X_test, y_train, y_test, 
            self.data_preparer.label_encoder, feature_names,
            df_original=self.data,  # Pasar datos originales para pesos por años
            train_indices=train_indices
        )
        
        # 8. Validar resultados
        successful_models = [name for name, metrics in results.items() if 'error' not in metrics]
        
        if not successful_models:
            print("❌ Ningún modelo se entrenó exitosamente")
            return False
        
        print(f"✅ Pipeline completado: {len(successful_models)} modelos entrenados")
        return True

    def collect_data(self):
        """Recolecta datos de FastF1"""
        self.collector.collect_data()
        self.data = self.collector.get_data()

    def preprocess_data(self):
        """Limpia y prepara los datos"""
        self.data = clean_data(self.data)

    def predict_next_race_positions(self):
        """Predice posiciones para la próxima carrera"""
        print("🎯 Prediciendo posiciones para próxima carrera...")

        predictor = SimplePositionPredictor()
        predictions_df = predictor.predict_positions_2025()
        predictor.show_realistic_predictions(predictions_df)

        # Guardar predicciones
        output_file = "app/models_cache/realistic_predictions_2025.csv"
        predictions_df.to_csv(output_file, index=False)
        print(f"💾 Predicciones guardadas: {output_file}")

        return predictions_df

    # ================== FP3 → Quali (2025) ==================
    def train_quali_from_fp3(self, year: int = 2025) -> bool:
        """Construye dataset de qualis recientes y entrena predictor basado SOLO en últimas qualis.
        Guarda el modelo y devuelve True si todo OK.

        Nota: Mantenemos el nombre por compatibilidad con el CLI."""
        try:
            years = [max(2024, int(year)-1), int(year)]
            print(f"🚀 Construyendo dataset de qualis para años {years}...")
            dataset = self.data_preparer.build_fp3_quali_dataset(year=years)
            if dataset is None or dataset.empty:
                print("❌ No se pudo construir el dataset de quali")
                return False

            print("🧠 Entrenando RecentQualiPredictor (sin FP3)...")
            predictor = RecentQualiPredictor()
            predictor.fit(dataset, n_recent=3)
            predictor.save("app/models_cache/quali_recent_model.pkl")
            print("✅ Modelo de qualis recientes guardado en app/models_cache/quali_recent_model.pkl")

            # Backtest opcional y guardado
            try:
                metrics = predictor.backtest(dataset)
                if metrics and metrics.get("events", 0) > 0:
                    dfm = pd.DataFrame([metrics])
                    dfm.to_csv("app/models_cache/quali_backtest_2025.csv", index=False)
                    print("📊 Backtest guardado: app/models_cache/quali_backtest_2025.csv")
                    print(f"   MAE={metrics.get('mae_s', float('nan')):.3f}s  MedAE={metrics.get('medae_s', float('nan')):.3f}s  Eventos={metrics.get('events', 0)}")
            except Exception as e:
                print(f"⚠️ Backtest omitido: {e}")

            return True
        except Exception as e:
            print(f"❌ Error entrenando quali desde FP3: {e}")
            return False

    def predict_quali_next_race(self) -> pd.DataFrame:
        """Predice quali de la próxima carrera usando SOLO últimas qualis.
        Genera app/models_cache/quali_predictions_latest.csv"""
        # Cargar modelo
        predictor = RecentQualiPredictor()
        if not predictor.load("app/models_cache/quali_recent_model.pkl"):
            print("ℹ️ No existe modelo de qualis recientes. Entrenando ahora...")
            ok = self.train_quali_from_fp3(year=2025)
            if not ok or not predictor.load("app/models_cache/quali_recent_model.pkl"):
                print("❌ No se pudo cargar/entrenar el modelo de qualis recientes")
                return pd.DataFrame()

        # Identificar próxima carrera
        nr = PREDICTION_CONFIG.get("next_race", {})
        year = int(nr.get("year", 2025))
        race_name = str(nr.get("race_name", ""))

        # Obtener round desde schedule
        try:
            import fastf1
            schedule = fastf1.get_event_schedule(year)
            round_num = int(schedule[schedule["EventName"] == race_name]["RoundNumber"].iloc[0])
        except Exception:
            round_num = int(nr.get("race_number", 0)) or None

        # Intentar recolectar info del mismo evento para usar SQ/S si existen
        event_df = pd.DataFrame()
        try:
            from app.data.collectors.fastf1_collector import FastF1Collector
            event_range = [{"year": year, "race_name": race_name, "round_number": round_num or 0}]
            ev_collector = FastF1Collector(event_range)
            ev_collector.collect_data()
            ev = ev_collector.get_data()
            if ev is not None and not ev.empty:
                cols = [c for c in ["driver", "sq_best_time", "sprint_position"] if c in ev.columns]
                if cols:
                    event_df = ev[cols].copy()
        except Exception as e:
            print(f"ℹ️ No se pudo recolectar info de sprint/SQ del evento: {e}")

        # Meta y predicción con refinamiento por sprint/SQ si está disponible
        meta = {"year": year, "race_name": race_name, "round": round_num}
        preds = predictor.predict_next_event(meta, event_df=event_df if not event_df.empty else None)

        # Guardar CSV
        out_path = "app/models_cache/quali_predictions_latest.csv"
        preds.to_csv(out_path, index=False)
        print(f"💾 Predicciones de quali guardadas: {out_path}")

        # Mostrar Top 10
        try:
            head = preds.head(10)
            print("\n🏁 Top 10 Predicción Quali")
            for _, r in head.iterrows():
                print(f"P{int(r['pred_rank']):<2} {r['driver']:<4} {str(r['team'])[:16]:<16} {r['pred_best_quali_lap']}")
        except Exception:
            pass

        return preds

    def _load_cached_data(self):
        """Carga datos desde cache"""
        cache_file = "app/models_cache/cached_data.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                self.data = pickle.load(f)
            print("📦 Datos cargados desde cache")
            return True
        return False

    def _save_cached_data(self):
        """Guarda datos en cache"""
        cache_dir = "app/models_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "cached_data.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"💾 Datos guardados en cache: {cache_file}")

    def _save_dataset_before_training(self):
        """Guarda el dataset en CSV antes del entrenamiento"""
        if self.data is None or len(self.data) == 0:
            print("⚠️  No hay datos para guardar")
            return
        
        # Crear directorio si no existe
        cache_dir = "app/models_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        
        # Generar nombre de archivo con timestamp
        csv_file = os.path.join(cache_dir, f"dataset_before_training_latest.csv")
        
        try:
            # Convertir a DataFrame si no lo es ya
            if isinstance(self.data, pd.DataFrame):
                df = self.data.copy()
            else:
                df = pd.DataFrame(self.data)
            
            # Guardar como CSV
            df.to_csv(csv_file, index=False)
            print(f"📊 Dataset guardado antes del entrenamiento: {csv_file}")
            print(f"   📈 Forma del dataset: {df.shape}")
            print(f"   📋 Columnas: {list(df.columns)}")
            
            # También guardar una versión sin timestamp para referencia
            latest_file = os.path.join(cache_dir, "dataset_before_training_latest.csv")
            df.to_csv(latest_file, index=False)
            print(f"📊 Versión latest guardada: {latest_file}")
            
        except Exception as e:
            print(f"❌ Error guardando dataset: {e}")
            # Intentar guardar información básica
            try:
                info_file = os.path.join(cache_dir, f"dataset_info_latest.txt")
                with open(info_file, 'w') as f:
                    f.write(f"Dataset info:\n")
                    f.write(f"Type: {type(self.data)}\n")
                    f.write(f"Length: {len(self.data)}\n")
                    if hasattr(self.data, 'shape'):
                        f.write(f"Shape: {self.data.shape}\n")
                    if hasattr(self.data, 'columns'):
                        f.write(f"Columns: {list(self.data.columns)}\n")
                print(f"📝 Info del dataset guardada: {info_file}")
            except:
                print("❌ No se pudo guardar información del dataset")

    def _save_dataset_after_feature_engineering(self, X_train, X_test, y_train, y_test):
        """Guarda el dataset después del feature engineering y limpieza"""
        try:
            # Crear directorio si no existe
            cache_dir = "app/models_cache"
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Combinar datos de entrenamiento y test
            X_combined = pd.concat([X_train, X_test], ignore_index=True)
            y_combined = pd.concat([y_train, y_test], ignore_index=True)
            
            # Crear DataFrame final con features + target (sin columna 'split') y con 'year' si está disponible
            df_processed = X_combined.copy()
            # Intentar añadir 'year' como primera columna para legibilidad
            try:
                train_years = getattr(self.data_preparer, 'train_years', None)
                test_years = getattr(self.data_preparer, 'test_years', None)
                if train_years is not None and test_years is not None:
                    years_series = pd.concat([train_years, test_years], ignore_index=True)
                    if len(years_series) == len(df_processed):
                        df_processed.insert(0, 'year', years_series.reset_index(drop=True))
            except Exception:
                pass
            df_processed['target_position'] = y_combined
            
            # Guardar archivos
            csv_file = os.path.join(cache_dir, f"dataset_after_feature_engineering_latest.csv")
            df_processed.to_csv(csv_file, index=False)
            
            
            print(f"🧠 Dataset procesado guardado: {csv_file}")
            print(f"   📈 Forma final (features + target): {df_processed.shape}")
            print(f"   🎯 Features finales:({len(X_combined.columns)}) {list(X_combined.columns)}")
            print(f"   📊 Split: {len(X_train)} train + {len(X_test)} test = {len(df_processed)} total")
            
        except Exception as e:
            print(f"❌ Error guardando dataset procesado: {e}")
            try:
                # Información básica como fallback
                info_file = os.path.join(cache_dir, f"processed_dataset_info_latest.txt")
                with open(info_file, 'w') as f:
                    f.write(f"Processed Dataset Info:\n")
                    f.write(f"X_train shape: {X_train.shape}\n")
                    f.write(f"X_test shape: {X_test.shape}\n")
                    f.write(f"y_train length: {len(y_train)}\n")
                    f.write(f"y_test length: {len(y_test)}\n")
                    f.write(f"Features: {list(X_train.columns)}\n")
                print(f"📝 Info del dataset procesado guardada: {info_file}")
            except:
                print("❌ No se pudo guardar información del dataset procesado")

    # Utilidad: forzar descarga fresca de eventos específicos
    def force_download_events(self, events: list[dict], use_persistent_fastf1_cache: bool = True) -> pd.DataFrame:
        """
        Descarga fresca (ignorando PKL locales) para una lista de eventos [{year, race_name, round_number?}].
        Devuelve DataFrame combinado de esos eventos.
        """
        cache_dir = "app/models_cache/fastf1"
        collector = FastF1Collector(events, force_refresh=True, fastf1_cache_dir=(cache_dir if use_persistent_fastf1_cache else None))
        collector.collect_data(force_refresh=True)
        df = collector.get_data()
        print(f"✅ Descarga forzada completada: {len(df)} filas")
        return df

    # ================== Orquestación end-to-end (opción 6) ==================
    def predict_race_from_quali_grid(self, beta: float | None = None) -> pd.DataFrame:
        """Predice la carrera usando la grilla (real si existe, si no la predicha por quali).
        - beta: peso de la grilla en la mezcla (0..1)."""
        print("🏎️ Generando predicción de carrera combinando modelo + grilla...")

        # 1) Predicción base de carrera (modelo entrenado)
        sp = SimplePositionPredictor()
        base = sp.predict_positions_2025()
        base = base.rename(columns={
            "final_position": "base_rank",
            "predicted_position": "base_score",
        })
        base = base[["driver", "team", "base_rank", "base_score"]].copy()

        # 2) Obtener grilla: intentar real del evento, luego predicha por quali
        nr = PREDICTION_CONFIG.get("next_race", {})
        year = int(nr.get("year", 2025))
        race_name = str(nr.get("race_name", ""))
        grid_df = None
        try:
            # recolectar evento y buscar grid_position de Q
            event_range = [{"year": year, "race_name": race_name, "round_number": nr.get("race_number", 0)}]
            evc = FastF1Collector(event_range)
            evc.collect_data()
            ev = evc.get_data()
            if ev is not None and not ev.empty and "grid_position" in ev.columns:
                tmp = ev[["driver", "grid_position"]].dropna().copy()
                if not tmp.empty:
                    tmp["grid_position"] = pd.to_numeric(tmp["grid_position"], errors="coerce")
                    grid_df = tmp
        except Exception:
            grid_df = None

        if grid_df is None or grid_df.empty:
            # usar quali predicha
            try:
                quali_path = "app/models_cache/quali_predictions_latest.csv"
                q = pd.read_csv(quali_path)
                if not q.empty:
                    grid_df = q[["driver", "pred_rank"]].rename(columns={"pred_rank": "grid_position"})
            except Exception:
                grid_df = None

        if grid_df is None or grid_df.empty:
            print("⚠️ No hay grilla disponible (real ni predicha). Devuelvo predicción base de carrera.")
            out = base.copy()
            out = out.sort_values("base_score", ascending=True).reset_index(drop=True)
            out["final_position"] = range(1, len(out) + 1)
            out.rename(columns={"base_score": "predicted_position"}, inplace=True)
            out = out[["final_position", "driver", "team", "predicted_position"]]
            return out

        # 3) Mezclar
        df = base.merge(grid_df, on="driver", how="left")
        # si no hay grid para alguno, usar promedio
        if df["grid_position"].isna().any():
            df["grid_position"].fillna(df["grid_position"].median() if df["grid_position"].notna().any() else 10.5, inplace=True)
        df["grid_position"] = pd.to_numeric(df["grid_position"], errors="coerce").fillna(10.5)

        # beta desde config si no se pasa explícito
        if beta is None:
            try:
                beta = float(RACE_PREDICTION.get("grid_mix_beta", 0.40))
            except Exception:
                beta = 0.40
        beta = max(0.0, min(1.0, float(beta)))
        df["race_score"] = (1 - beta) * df["base_score"].astype(float) + beta * df["grid_position"].astype(float)
        df = df.sort_values("race_score", ascending=True).reset_index(drop=True)
        df["final_position"] = range(1, len(df) + 1)

        out = df[["final_position", "driver", "team", "base_score", "grid_position", "race_score"]].copy()
        out.rename(columns={"base_score": "model_position_score", "race_score": "predicted_position"}, inplace=True)

        # Guardar
        out_path = "app/models_cache/race_predictions_latest.csv"
        out.to_csv(out_path, index=False)
        print(f"💾 Predicción de carrera guardada: {out_path}")
        return out

    def train_and_predict_all(self) -> dict:
        """Entrena modelos (quali + carrera), predice quali y luego predice carrera usando grilla.
        Devuelve rutas de archivos generados."""
        print("🧩 Opción 6: Entrenar y predecir quali + carrera")
        artifacts = {}
        # 1) Entrenar quali (histórico recientes)
        ok_q = self.train_quali_from_fp3(year=2025)
        if not ok_q:
            print("⚠️ Entrenamiento de quali fallido o incompleto")
        artifacts["quali_model"] = "app/models_cache/quali_recent_model.pkl"

        # 2) Entrenar carrera (pipeline principal si hace falta)
        self.run()
        artifacts["race_models"] = "app/models_cache/"  # carpeta

        # 3) Predecir quali de la próxima
        qpred = self.predict_quali_next_race()
        artifacts["quali_predictions"] = "app/models_cache/quali_predictions_latest.csv"

        # 4) Predecir carrera usando grilla (real si hay; si no, predicha)
        rpred = self.predict_race_from_quali_grid(beta=None)
        artifacts["race_predictions"] = "app/models_cache/race_predictions_latest.csv"
        return artifacts

    def predict_all(self) -> dict:
        """predice quali y luego predice carrera usando grilla.
        Devuelve rutas de archivos generados."""
        artifacts = {}
        artifacts["quali_model"] = "app/models_cache/quali_recent_model.pkl"

        artifacts["race_models"] = "app/models_cache/"  # carpeta

        # 3) Predecir quali de la próxima
        qpred = self.predict_quali_next_race()
        artifacts["quali_predictions"] = "app/models_cache/quali_predictions_latest.csv"

        # 4) Predecir carrera usando grilla (real si hay; si no, predicha)
        rpred = self.predict_race_from_quali_grid(beta=None)
        artifacts["race_predictions"] = "app/models_cache/race_predictions_latest.csv"
        return artifacts

