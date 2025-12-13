from sklearn.model_selection import GroupKFold, KFold
import numpy as np
from app.config import DATA_IMPORTANCE

def build_groups(df_original, train_indices, X_train):
    groups = None
    if df_original is not None:
        try:
            if train_indices is not None:
                df_train = df_original.iloc[train_indices].copy()
            else:
                df_train = df_original.iloc[:len(X_train)].copy()
            df_train["race_id"] = df_train["year"].astype(str) + "_" + df_train["race_name"].astype(str)
            groups = df_train["race_id"].values
        except Exception:
            groups = None
    return groups


def get_cv_splitter(X_train, groups=None):
    if groups is not None and len(groups) == len(X_train):
        n_unique = len(np.unique(groups))
        n_splits = min(5, max(3, n_unique))
        print(f"📊 Usando GroupKFold con {n_splits} splits (grupo=race_id)")
        return GroupKFold(n_splits=n_splits)
    n_samples = len(X_train)
    n_splits = max(3, min(5, n_samples // 5))
    print(f"📊 Usando KFold con {n_splits} splits")
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)




def calculate_year_weights(df_with_years):
    if 'year' not in df_with_years.columns:
        print("⚠️ No se encontró columna 'year', usando pesos uniformes")
        return np.ones(len(df_with_years))

    year_weights_map = {
        2025: DATA_IMPORTANCE.get("2025_weight", 0.50),
        2024: DATA_IMPORTANCE.get("2024_weight", 0.25),
        2023: DATA_IMPORTANCE.get("2023_weight", 0.15),
        2022: DATA_IMPORTANCE.get("2022_weight", 0.10),
    }

    weights = df_with_years['year'].map(year_weights_map).fillna(0.05)
    weights = weights * len(weights) / weights.sum()

    print("📅 Pesos aplicados por año:")
    year_counts = df_with_years['year'].value_counts().sort_index()
    for year in sorted(year_counts.index):
        weight = year_weights_map.get(year, 0.05)
        count = year_counts[year]
        print(f"   {year}: {weight:.1%} peso × {count} muestras")
    return weights.values

