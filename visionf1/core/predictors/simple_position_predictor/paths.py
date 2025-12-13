MODEL_FILES = {
    "RandomForest": "visionf1/models_cache/randomforest_model.pkl",
    "XGBoost": "visionf1/models_cache/xgboost_model.pkl",
    "GradientBoosting": "visionf1/models_cache/gradientboosting_model.pkl",
}

AFTER_FE_PATH = "visionf1/models_cache/dataset_after_feature_engineering_latest.csv"
BEFORE_FE_PATH = "visionf1/models_cache/dataset_before_training_latest.csv"
CACHED_DATA_PKL = "visionf1/models_cache/cached_data.pkl"
FEATURE_NAMES_PKL = "visionf1/models_cache/feature_names.pkl"
INFERENCE_OUT = "visionf1/models_cache/inference_dataset_latest.csv"

MANIFEST_PATHS = [
    "visionf1/models_cache/inference_manifest.json",
    "/mnt/data/inference_manifest.json",
]

TRAINING_RESULTS_PKL = "visionf1/models_cache/training_results.pkl"
