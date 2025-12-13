from sklearn.model_selection import GridSearchCV, cross_val_score

from .hyperparameter_config import get_param_combinations
from .reporting import show_hyperparameter_search_details, analyze_parameter_importance
from .metrics_utils import build_metrics, detect_overfitting_patterns


def train_single_model_with_cv(name, model_config, X_train, X_test, y_train, y_test, cv_splitter, sample_weights=None, groups=None):
    model_class = model_config['model_class']
    base_model = model_class()

    print(f"🔧 Optimizando hiperparámetros para {name}...")

    fit_params = {}
    if hasattr(base_model.model, "early_stopping_rounds") and name != 'XGBoost':
        fit_params = {"eval_set": [(X_test, y_test)], "early_stopping_rounds": 10, "verbose": False}

    grid_search = GridSearchCV(
        base_model.model,
        model_config['param_grid'],
        cv=cv_splitter,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0,
    )

    combinations = get_param_combinations(model_config['param_grid'])
    print(f"🔍 Calculando {combinations} combinaciones de hiperparámetros...")

    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["sample_weight"] = sample_weights

    grid_search.fit(X_train, y_train, groups=groups, **fit_kwargs)

    show_hyperparameter_search_details(grid_search, name, analyze_parameter_importance)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"✅ Mejores parámetros: {best_params}")
    print(f"📊 Evaluando con Cross-Validation...")

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_splitter, scoring='neg_mean_squared_error', groups=groups)
    cv_mse_scores = -cv_scores; cv_mean = cv_mse_scores.mean(); cv_std = cv_mse_scores.std()

    metrics = build_metrics(best_model, X_train, X_test, y_train, y_test, cv_scores, cv_std, cv_mean)
    metrics['best_params'] = best_params

    detect_overfitting_patterns(metrics, name)

    optimized_model = model_class(); optimized_model.model = best_model
    return metrics, optimized_model
