MODEL_FILES = {
    "RandomForest": "models_cache/randomforest_model.pkl",
    "XGBoost": "models_cache/xgboost_model.pkl",
    "GradientBoosting": "models_cache/gradientboosting_model.pkl",
}

AFTER_FE_PATH = "models_cache/dataset_after_feature_engineering_latest.csv"
BEFORE_FE_PATH = "models_cache/dataset_before_training_latest.csv"
CACHED_DATA_PKL = "models_cache/cached_data.pkl"
FEATURE_NAMES_PKL = "models_cache/feature_names.pkl"
INFERENCE_OUT = "models_cache/inference_dataset_latest.csv"

MANIFEST_PATHS = [
    "models_cache/inference_manifest.json",
    "/mnt/data/inference_manifest.json",
]

TRAINING_RESULTS_PKL = "models_cache/training_results.pkl"
