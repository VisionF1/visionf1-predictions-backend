MODEL_FILES = {
    "RandomForest": "app/models_cache/randomforest_model.pkl",
    "XGBoost": "app/models_cache/xgboost_model.pkl",
    "GradientBoosting": "app/models_cache/gradientboosting_model.pkl",
}

AFTER_FE_PATH = "app/models_cache/dataset_after_feature_engineering_latest.csv"
BEFORE_FE_PATH = "app/models_cache/dataset_before_training_latest.csv"
CACHED_DATA_PKL = "app/models_cache/cached_data.pkl"
FEATURE_NAMES_PKL = "app/models_cache/feature_names.pkl"
INFERENCE_OUT = "app/models_cache/inference_dataset_latest.csv"

MANIFEST_PATHS = [
    "app/models_cache/inference_manifest.json",
    "/mnt/data/inference_manifest.json",
]

TRAINING_RESULTS_PKL = "app/models_cache/training_results.pkl"
