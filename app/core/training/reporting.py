import pandas as pd


def is_pre_race_safe(name: str):
    n = (name or '').lower()
    unsafe_tokens = [
        'race_position', 'final_position', 'grid_position', 'grid_to_race_change',
        'quali_vs_race_delta', 'points_efficiency', 'fastest_lap', 'status',
        'race_best_lap', 'lap_time_std', 'lap_time_consistency'
    ]
    if n == 'fp3_best_time':
        return True, 'fp3 (pre-quali)'
    if any(tok in n for tok in unsafe_tokens):
        return False, 'contiene métrica de carrera/resultado'
    safe_prefixes = [
        'fp1_', 'fp2_', 'fp3_', 'session_', 'weather_', 'team_avg_position_',
        'driver_', 'team_', 'expected_grid_position', 'points_last_3',
        'avg_position_last_3', 'avg_quali_last_3', 'overtaking_ability',
        'team_track_avg_position', 'driver_track_avg_position', 'sector_',
        'heat_index', 'temp_deviation_from_ideal', 'weather_difficulty_index',
        'total_laps'
    ]
    if any(n.startswith(p) for p in safe_prefixes):
        return True, 'histórico/prácticas/clima'
    if 'quali_position' in n:
        return False, 'quali directa (del evento)'
    return True, 'genérica'


def audit_features(feature_names, target_name=None, prefix="Entrenamiento"):
    if not feature_names:
        print("⚠️  Sin feature_names para auditar")
        return
    print(f"\n🧾 {prefix}: FEATURES USADAS ({len(feature_names)})")
    if target_name:
        print(f"   🎯 Target: {target_name}")
    safe, unsafe = [], []
    for f in feature_names:
        is_safe, reason = is_pre_race_safe(f)
        (safe if is_safe else unsafe).append((f, reason))
    if unsafe:
        print("   🚫 Sospecha post-race (revisar):")
        for name, reason in unsafe:
            print(f"      - {name}  [{reason}]")
    print("   ✅ Pre-race (ok):")
    for name, _ in safe:
        print(f"      - {name}")
    try:
        out_path = "app/models_cache/feature_audit.txt"
        rows = [f"Target: {target_name}\n\n", "UNSAFE (post-race sospechoso):\n"]
        rows += [f"- {n} [{r}]\n" for n, r in unsafe]
        rows += ["\nSAFE (pre-race):\n"]
        rows += [f"- {n}\n" for n, _ in safe]
        with open(out_path, 'w') as f:
            f.writelines(rows)
        print(f"📄 Auditoría de features guardada: {out_path}")
    except Exception:
        pass


def show_model_results(name, metrics):
    if 'error' in metrics:
        print(f"\n❌ {name}: Error - {metrics['error']}")
        return
    print(f"\n📊 RESULTADOS DETALLADOS - {name}")
    print("-" * 40)
    print(f"🔄 Cross-Validation MSE: {metrics['cv_mse_mean']:.4f} ± {metrics['cv_mse_std']:.4f}")
    print(f"🏋️  Train MSE: {metrics['train_mse']:.4f}")
    print(f"🎯 Test MSE:  {metrics['test_mse']:.4f}")
    print(f"📈 Test R²:   {metrics['test_r2']:.4f}")
    print(f"🔗 Kendall’s tau (train): {metrics['kendall_tau_train']:.3f}")
    print(f"🔗 Kendall’s tau (test):  {metrics['kendall_tau_test']:.3f}")
    print(f"🔗 Spearman rho (test):   {metrics['spearman_rho_test']:.3f}")
    over = metrics['overfitting_score']
    if over < 1.1:
        print(f"✅ Overfitting Score: {over:.2f} (Bueno)")
    elif over < 1.5:
        print(f"⚠️  Overfitting Score: {over:.2f} (Moderado)")
    else:
        print(f"🚨 Overfitting Score: {over:.2f} (Alto)")
    if metrics.get('best_params', 'default') != 'default':
        print(f"🔧 Mejores parámetros: {metrics['best_params']}")


def show_detailed_comparison(results):
    print(f"\n{'='*80}")
    print("🏆 COMPARACIÓN DETALLADA DE MODELOS CON CROSS-VALIDATION")
    print(f"{'='*80}")
    print(f"{'Modelo':<15} {'CV MSE':<12} {'Test MSE':<10} {'Test R²':<8} {'Kendalls Tau':<12} {'Overfitting':<12} {'Estado'}")
    print("-" * 80)
    best_cv = float('inf')
    best_model = None
    for name, m in results.items():
        if 'error' in m:
            print(f"{name:<15} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8} {'ERROR':<12} {'❌'}")
            continue
        cv_score = m['cv_mse_mean']; test_mse = m['test_mse']; test_r2 = m['test_r2']
        over = m['overfitting_score']; kendall_tau = m['kendall_tau_test']
        status = "✅ Bueno" if over < 1.1 else ("⚠️  Moderado" if over < 1.5 else "🚨 Overfitting")
        print(f"{name:<15} {cv_score:<12.4f} {test_mse:<10.4f} {test_r2:<8.4f} {kendall_tau:<12.2f} {over:<12.2f} {status}")
        if cv_score < best_cv and over < 1.3:
            best_cv = cv_score; best_model = name
    if best_model:
        print(f"\n🏆 MEJOR MODELO: {best_model}\n   📊 CV MSE: {best_cv:.4f}\n   🎯 Sin overfitting significativo")
    else:
        print("\n⚠️  ADVERTENCIA: Todos los modelos muestran problemas de overfitting\n   💡 Considera reducir la complejidad o conseguir más datos")


def show_hyperparameter_search_details(grid_search, model_name, analyze_parameter_importance_fn):
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_results = results_df.nlargest(5, 'mean_test_score')
    print("🏆 TOP 5 MEJORES COMBINACIONES:")
    for i, (idx, row) in enumerate(top_results.iterrows(), 1):
        score = -row['mean_test_score']; std = row['std_test_score']; params = row['params']
        print(f"\n   {i}. Score: {score:.4f} (±{std:.4f})\n      Parámetros: {params}")
    all_scores = -results_df['mean_test_score']
    print("\n📊 ESTADÍSTICAS DE LA BÚSQUEDA:")
    print(f"   🎯 Mejor score: {all_scores.min():.4f}")
    print(f"   📈 Score promedio: {all_scores.mean():.4f}")
    print(f"   📉 Peor score: {all_scores.max():.4f}")
    print(f"   📏 Desviación estándar: {all_scores.std():.4f}")
    print(f"   🔄 Mejora vs promedio: {((all_scores.mean() - all_scores.min()) / all_scores.mean() * 100):.1f}%")
    analyze_parameter_importance_fn(results_df, model_name)


def analyze_parameter_importance(results_df, model_name):
    print("\n🔍 ANÁLISIS DE IMPORTANCIA DE PARÁMETROS:")
    param_columns = {}
    for _, row in results_df.iterrows():
        for p, v in row['params'].items():
            param_columns.setdefault(p, []).append(v)
    scores = -results_df['mean_test_score']
    for param_name, param_values in param_columns.items():
        unique_values = list(set(param_values))
        if len(unique_values) <= 1:
            continue
        print(f"\n   📊 {param_name}:")
        param_scores = {}
        for value in unique_values:
            mask = [v == value for v in param_values]
            value_scores = scores[mask]
            if len(value_scores) > 0:
                param_scores[value] = {
                    'mean': value_scores.mean(),
                    'std': value_scores.std(),
                    'count': len(value_scores),
                }
        sorted_params = sorted(param_scores.items(), key=lambda x: x[1]['mean'])
        for value, stats in sorted_params:
            print(f"      {value}: {stats['mean']:.4f} (±{stats['std']:.4f}) [{stats['count']} pruebas]")
        best_value = sorted_params[0][0]; worst_value = sorted_params[-1][0]
        improvement = sorted_params[-1][1]['mean'] - sorted_params[0][1]['mean']
        print(f"      ✅ Mejor: {best_value} | ❌ Peor: {worst_value} | 📈 Diferencia: {improvement:.4f}")
