from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau, spearmanr


def calculate_overfitting_score(train_mse, test_mse):
    if train_mse == 0:
        return float('inf') if test_mse > 0 else 0
    return test_mse / train_mse


def calculate_overfitting_level(metrics):
    weights = {
        'overfitting_score': 0.4,
        'generalization_gap': 0.3,
        'cv_stability': 0.2,
        'train_test_r2_gap': 0.1,
    }
    norm_overfitting = min(metrics['overfitting_score'] - 1.0, 1.0)
    norm_gen_gap = min(metrics['generalization_gap'] / 10.0, 1.0)
    norm_cv_stability = min(metrics['cv_stability'] / 0.2, 1.0)
    norm_r2_gap = min(metrics['train_test_r2_gap'] / 0.2, 1.0)
    combined = (
        weights['overfitting_score'] * norm_overfitting +
        weights['generalization_gap'] * norm_gen_gap +
        weights['cv_stability'] * norm_cv_stability +
        weights['train_test_r2_gap'] * norm_r2_gap
    )
    if combined < 0.3:
        return "🟢 Bajo (Bueno)"
    elif combined < 0.6:
        return "🟡 Moderado"
    else:
        return "🔴 Alto (Crítico)"


def detect_overfitting_patterns(metrics, model_name):
    print(f"\n🔍 ANÁLISIS DE OVERFITTING - {model_name}:")
    gen_gap = metrics['generalization_gap']
    if gen_gap > 5.0:
        print(f"   🚨 Gap de generalización alto: {gen_gap:.3f} (Train vs Test)")
    elif gen_gap > 2.0:
        print(f"   ⚠️  Gap de generalización moderado: {gen_gap:.3f}")
    else:
        print(f"   ✅ Gap de generalización bueno: {gen_gap:.3f}")
    cv_stability = metrics['cv_stability']
    if cv_stability > 0.15:
        print(f"   🚨 CV inestable: {cv_stability:.3f} (alta variabilidad)")
    elif cv_stability > 0.10:
        print(f"   ⚠️  CV moderadamente estable: {cv_stability:.3f}")
    else:
        print(f"   ✅ CV estable: {cv_stability:.3f}")
    r2_gap = metrics['train_test_r2_gap']
    if r2_gap > 0.15:
        print(f"   🚨 R² gap alto: {r2_gap:.3f} (sobreajuste)")
    elif r2_gap > 0.08:
        print(f"   ⚠️  R² gap moderado: {r2_gap:.3f}")
    else:
        print(f"   ✅ R² gap bueno: {r2_gap:.3f}")
    print(f"   📊 Nivel de overfitting: {calculate_overfitting_level(metrics)}")
    if any([gen_gap > 3.0, cv_stability > 0.12, r2_gap > 0.10]):
        print("   💡 RECOMENDACIONES:")
        if gen_gap > 3.0:
            print("      - Incrementar regularización\n      - Reducir complejidad del modelo")
        if cv_stability > 0.12:
            print("      - Aumentar datos de entrenamiento\n      - Usar ensemble methods")
        if r2_gap > 0.10:
            print("      - Feature selection más agresiva\n      - Early stopping más temprano")


def build_metrics(model, X_train, X_test, y_train, y_test, cv_scores, cv_std, cv_mean):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    tau_train, _ = kendalltau(y_train, y_pred_train)
    tau_test, _ = kendalltau(y_test, y_pred_test)
    rho_train, _ = spearmanr(y_train, y_pred_train)
    rho_test, _ = spearmanr(y_test, y_pred_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)

    metrics = {
        'cv_mse_mean': cv_mean,
        'cv_mse_std': cv_std,
        'cv_scores': (-cv_scores).tolist(),
        'train_mse': train_mse,
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_mse': test_mse,
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'test_r2': r2_score(y_test, y_pred_test),
        'kendall_tau_train': tau_train,
        'kendall_tau_test': tau_test,
        'spearman_rho_train': rho_train,
        'spearman_rho_test': rho_test,
        'overfitting_score': calculate_overfitting_score(train_mse, test_mse),
        'generalization_gap': test_mse - train_mse,
        'cv_stability': cv_std / cv_mean if cv_mean > 0 else float('inf'),
        'train_test_r2_gap': r2_score(y_train, y_pred_train) - r2_score(y_test, y_pred_test),
    }
    return metrics

