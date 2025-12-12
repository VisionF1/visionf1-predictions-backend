import pickle
from pathlib import Path
import json

INFERENCE_MANIFEST_PATH = Path("app/models_cache/inference_manifest.json")
CATEGORICAL_COLS = ["driver", "team", "race_name", "circuit_type"]


def save_model(name, model):
    model_path = f"app/models_cache/{name.lower()}_model.pkl"
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Modelo optimizado guardado: {model_path}")
    except Exception as e:
        print(f"❌ Error guardando modelo: {e}")


def save_metadata(label_encoder, feature_names):
    print("💾 Guardando metadata del modelo!!!!!!!!!!!!!...")
    print(f"   - Features: {len(feature_names) if feature_names else 0}")
    print(f"   - Label Encoder: {type(label_encoder).__name__ if label_encoder else 'None'}")
    try:
        Path("app/models_cache").mkdir(parents=True, exist_ok=True)
        with open("app/models_cache/label_encoder.pkl", 'wb') as f:
            pickle.dump(label_encoder, f)
        with open("app/models_cache/feature_names.pkl", 'wb') as f:
            pickle.dump(feature_names, f)
        print("✅ Metadata guardada")
    except Exception as e:
        print(f"❌ Error guardando metadata: {e}")


def save_training_results(results, label_encoder, feature_names):
    try:
        results_file = "app/models_cache/training_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 Métricas de entrenamiento guardadas: {results_file}")
        if results:
            try:
                save_inference_manifest(feature_names, results, selection_metric="kendall")
            except Exception as e:
                print(f"⚠️ No se pudo guardar manifiesto de inferencia: {e}")
        save_readable_summary(results)
    except Exception as e:
        print(f"❌ Error guardando métricas: {e}")


def save_readable_summary(results):
    try:
        summary_file = "app/models_cache/model_comparison.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPARACIÓN DE MODELOS - MÉTRICAS DE ENTRENAMIENTO\n")
            f.write("="*60 + "\n\n")
            for name, m in results.items():
                if 'error' in m:
                    f.write(f"{name}: ERROR - {m['error']}\n\n")
                    continue
                f.write(f"{name}:\n")
                f.write(f"  CV MSE: {m.get('cv_mse_mean', 'N/A'):.4f}\n")
                f.write(f"  Test MSE: {m.get('test_mse', 'N/A'):.4f}\n")
                f.write(f"  Test R²: {m.get('test_r2', 'N/A'):.4f}\n")
                f.write(f"  Overfitting Score: {m.get('overfitting_score', 'N/A'):.2f}\n")
                f.write(f"  Mejor Parámetros: {m.get('best_params', 'N/A')}\n\n")
        print(f"📄 Resumen legible guardado: {summary_file}")
    except Exception as e:
        print(f"❌ Error guardando resumen: {e}")


def save_inference_manifest(feature_names, results, selection_metric="kendall"):
    best = None
    if selection_metric == "kendall":
        candidates = []
        for name, m in results.items():
            if "error" in m:
                continue
            score = (m.get("kendall_tau_test", -1.0) or -1.0) - 0.05 * max(0.0, m.get("overfitting_score", 1.0) - 1.0)
            candidates.append((score, name))
        if candidates:
            best = sorted(candidates, reverse=True)[0][1]
    else:
        best_cv = float('inf'); best = None
        for name, m in results.items():
            if "error" in m:
                continue
            if m.get("overfitting_score", 2.0) >= 1.3:
                continue
            if m.get("cv_mse_mean", float('inf')) < best_cv:
                best = name; best_cv = m["cv_mse_mean"]

    encoders = {}
    for col in ["driver", "team", "race_name", "circuit_type"]:
        path = Path(f"app/models_cache/{col}_encoder.pkl")
        encoders[col] = str(path) if path.exists() else None

    manifest = {
        "feature_names": list(feature_names or []),
        "categorical_cols": ["driver", "team", "race_name", "circuit_type"],
        "encoders": encoders,
        "best_model_name": best,
    }
    INFERENCE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INFERENCE_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"💾 Manifiesto de inferencia guardado: {INFERENCE_MANIFEST_PATH}")
