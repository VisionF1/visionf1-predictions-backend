from app.core.predictors.models.random_forest import RandomForestPredictor
from app.core.predictors.models.xgboost_model import XGBoostPredictor
from app.core.predictors.models.gradient_boosting import GradientBoostingPredictor

def initialize_models_with_hyperparams():
    return {
        'RandomForest': {
            'model_class': RandomForestPredictor,
            'param_grid': {
            'n_estimators': [50, 100],
            'max_depth': [2, 3],
            'min_samples_split': [20, 30],
            'min_samples_leaf': [10, 15],
            'max_features': ['sqrt', 0.5],
            'bootstrap': [True],
            'oob_score': [True]
            }
        },
        'XGBoost': {
            'model_class': XGBoostPredictor,
            'param_grid': {
            'n_estimators': [200, 400],
            'max_depth': [2, 3],
            'learning_rate': [0.03, 0.06],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
            'reg_alpha': [2.0, 4.0],
            'reg_lambda': [2.0, 4.0],
            'min_child_weight': [10, 20]
            }
        },
        'GradientBoosting': {
            'model_class': GradientBoostingPredictor,
            'param_grid': {
            'n_estimators': [30, 50],
            'max_depth': [2, 3],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.6, 0.8],
            'min_samples_split': [20, 30],
            'min_samples_leaf': [10, 14],
            'max_features': ['sqrt', 0.5],
            'validation_fraction': [0.1]
            }
        }
    }


def get_param_combinations(param_grid):
    total = 1
    for values in param_grid.values():
        total *= len(values) if isinstance(values, (list, tuple)) else 1
    return total

