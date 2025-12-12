import warnings
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pickle

from app.core.features.advanced_feature_engineer import AdvancedFeatureEngineer

warnings.filterwarnings('ignore')


class EnhancedDataPreparer:
    def __init__(self, quiet: bool = False, de_emphasize_team: bool = True, team_deemphasis_factor: float = 0.4):
        self.quiet = quiet
        self.feature_engineer = AdvancedFeatureEngineer(quiet=self.quiet)
        self.label_encoder = LabelEncoder()  # (legacy, no lo usamos directamente)
        self.feature_names: List[str] = []
        self.de_emphasize_team = de_emphasize_team
        self.team_deemphasis_factor = max(0.0, min(1.0, team_deemphasis_factor))

    # ---------- utils ----------
    def _log(self, msg: str):
        if not self.quiet:
            print(msg)

    def _load_encoder(self, col: str) -> Optional[LabelEncoder]:
        p = Path(f"app/models_cache/{col}_encoder.pkl")
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def _save_encoder(self, col: str, le: LabelEncoder):
        p = Path(f"app/models_cache/{col}_encoder.pkl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(le, f)

    def _transform_with_encoder(self, le: LabelEncoder, series: pd.Series, unknown_index: int = 0) -> np.ndarray:
        """Transforma usando un LabelEncoder sin modificar sus clases.
        Cualquier valor desconocido se mapea a unknown_index (por defecto 0)."""
        vals = series.astype(str).fillna("")
        classes = list(le.classes_)
        mapping = {c: i for i, c in enumerate(classes)}
        # si unknown_index está fuera de rango, forzamos 0
        if not (0 <= unknown_index < len(classes)):
            unknown_index = 0
        return np.array([mapping.get(v, unknown_index) for v in vals], dtype=int)

    def _fit_or_transform_categorical(self, df: pd.DataFrame, col: str, inference: bool) -> pd.DataFrame:
        if col not in df.columns:
            return df

        df = df.dropna(subset=[col]).copy()
        df[col] = df[col].astype(str)

        if inference:
            le = self._load_encoder(col)
            if le is None:
                # fallback: encoder ad-hoc sólo para esta predicción (no se guarda)
                le = LabelEncoder().fit(df[col])
            df[f"{col}_encoded"] = self._transform_with_encoder(le, df[col], unknown_index=0)
        else:
            # ENTRENAMIENTO: ajustamos y guardamos
            le = self._load_encoder(col)
            if le is None:
                le = LabelEncoder().fit(df[col])
            else:
                # extender clases de forma estable SIN reordenar indices existentes:
                # LabelEncoder ordena alfabéticamente, así que NO vamos a "extender"
                # para no desplazar índices. Simplemente re-fit con todas las clases
                # actuales del df + las ya existentes, pero mantenemos el mismo orden.
                # Para lograrlo, congelamos el orden previo y sólo añadimos nuevas
                # al final.
                prev = list(le.classes_)
                new_vals = sorted(set(df[col].unique()) - set(prev))
                if new_vals:
                    all_classes = np.array(prev + new_vals, dtype=object)
                    # Creamos un "pseudo" encoder con ese orden fijo:
                    le = LabelEncoder()
                    le.classes_ = all_classes

            # transform y guardado
            df[f"{col}_encoded"] = self._transform_with_encoder(le, df[col], unknown_index=0)
            self._save_encoder(col, le)

        return df

    def _process_weather_features(self, df: pd.DataFrame):
        defaults = {
            'session_air_temp': 25.0,
            'session_track_temp': 35.0,
            'session_humidity': 60.0,
            'session_rainfall': 0.0,
        }
        for c, v in defaults.items():
            if c not in df.columns:
                df[c] = v
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(v)

        if 'heat_index' not in df.columns:
            air = df['session_air_temp']
            track = df['session_track_temp']
            hum = df['session_humidity']
            hi = (0.6 * air + 0.4 * track) / 100.0 - (hum - 50.0) * 0.001
            df['heat_index'] = np.clip(hi, 0.0, 1.0)

        if 'weather_difficulty_index' not in df.columns:
            df['weather_difficulty_index'] = (df['session_humidity'] / 100.0) + (df['session_rainfall'] * 3.0)

        return df

    # ---------- main ----------
    def prepare_enhanced_features(self, df: pd.DataFrame, inference: bool = False):
        """Genera el set de features pre-race seguro. En inference reutiliza encoders guardados."""
        df = df.copy()

        # Feature engineering pre-race
        df = self.feature_engineer.create_circuit_compatibility_features(df)
        df = self.feature_engineer.create_momentum_features(df)  # sólo histórico (usa shift(1))
        df = self._process_weather_features(df)
        df = self.feature_engineer.create_weather_performance_features(df)

        # Categóricas (usando encoders persistentes si inference=True)
        df = self._fit_or_transform_categorical(df, "team", inference)
        df = self._fit_or_transform_categorical(df, "race_name", inference)
        df = self._fit_or_transform_categorical(df, "driver", inference)
        df = self._fit_or_transform_categorical(df, "circuit_type", inference)

        # Selección final de features pre-race safe
        requested_features = [
            'driver_encoded', 'team_encoded', 'race_name_encoded', 'year',
            'driver_competitiveness', 'team_competitiveness',
            'driver_skill_factor', 'team_strength_factor', 'driver_team_synergy',
            'driver_weather_skill', 'overtaking_ability',
            'points_last_3',
            'session_air_temp', 'session_track_temp', 'session_humidity', 'session_rainfall',
            'heat_index', 'weather_difficulty_index',
            'circuit_type_encoded', 'driver_avg_points_in_rain',
            'driver_avg_points_in_dry', 'driver_rain_dry_delta'
        ]

        available = [f for f in requested_features if f in df.columns]
        X = df[available].copy()

        # Target si está presente (sólo entreno)
        y = None
        if not inference:
            for col in ['final_position', 'race_position', 'position']:
                if col in df.columns:
                    y = df[col].copy()
                    break

        self.feature_names = list(X.columns)
        return X, y, None, self.feature_names

    def prepare_training_data(self, df: pd.DataFrame):
        X, y, label_encoder, feature_names = self.prepare_enhanced_features(df, inference=False)
        if X is None or y is None:
            return None, None, None, None, None

        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(range(len(X)), test_size=0.2, random_state=42, shuffle=True)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        return X_train, X_test, y_train, y_test, feature_names




    # ---------- FP3 → Quali dataset (2025) ----------
    def build_fp3_quali_dataset(self, year: int | list[int] = 2025) -> pd.DataFrame:
        """Construye dataset FP3→Quali para uno o varios años y lo guarda en cache.

        Columnas clave del output:
        - year, round, race_name, weekend_key
        - driver, team
        - fp3_best_time, fp3_laps_count
        - quali_best_time
        - ratio_fp3_quali, delta_fp3_quali_s
        - event_delta_median_s (proxy evolución)
        """
        from app.core.utils.race_range_builder import RaceRangeBuilder
        from app.data.collectors.fastf1_collector import FastF1Collector
        from app.data.preprocessors.data_cleaner import clean_data
        import fastf1

        # 1) Construir rango de carreras para los años solicitados
        years = year
        if isinstance(years, (int, float, str)):
            try:
                years = [int(years)]
            except Exception:
                years = [2025]
        years = [int(y) for y in years]
        cfg = {"years": years, "max_races_per_year": 30}
        rr = RaceRangeBuilder().build_race_range(cfg)
        collector = FastF1Collector(rr)
        collector.collect_data()
        df = collector.get_data()
        if df is None or df.empty:
            return pd.DataFrame()

        # 2) Limpieza básica (no eliminar filas por columnas genéricas)
        # Evitar usar clean_data aquí porque elimina filas por 'best_lap_time' o 'clean_air_pace'
        # que no aplican a quali; nos quedamos con deduplicación ligera más abajo
        df = df.copy()

        # 3) Round mapping desde schedule (por (year, race_name))
        name_to_round = {}
        try:
            for y in years:
                try:
                    sched = fastf1.get_event_schedule(int(y))
                    for _, r in sched.iterrows():
                        rn = int(r.get("RoundNumber")) if pd.notna(r.get("RoundNumber")) else None
                        evn = str(r.get("EventName"))
                        if rn is not None and evn:
                            name_to_round[(int(y), evn)] = rn
                except Exception:
                    continue
        except Exception:
            name_to_round = {}

        # 4) Selección/derivación de columnas
        out = df.copy()
        if "year" in out.columns:
            out = out[out["year"].isin(years)]
        # columnas mínimas
        for c in ["driver", "team", "race_name", "year"]:
            if c not in out.columns:
                out[c] = None
        # round + weekend_key
        def _map_round(row):
            try:
                return name_to_round.get((int(row["year"]), str(row["race_name"])) , np.nan)
            except Exception:
                return np.nan
        out["round"] = out.apply(_map_round, axis=1)
        out["weekend_key"] = out.apply(lambda r: f"{int(r['year'])}_{str(r['race_name'])}", axis=1)
        # fp3/quali best
        out["fp3_best_time"] = pd.to_numeric(out.get("fp3_best_time", np.nan), errors="coerce")
        # Derivar quali_best_time si falta
        if "quali_best_time" not in out.columns or out["quali_best_time"].isna().all():
            qcols = []
            for c in ("q1_time", "q2_time", "q3_time", "quali_best_lap_from_laps"):
                if c in out.columns:
                    qcols.append(pd.to_numeric(out[c], errors="coerce"))
            if qcols:
                out["quali_best_time"] = pd.concat(qcols, axis=1).min(axis=1)
        out["quali_best_time"] = pd.to_numeric(out.get("quali_best_time", np.nan), errors="coerce")

        # 5) Filtrar válidos SOLO por quali (permitir fp3 faltante)
        out = out.dropna(subset=["quali_best_time"]).copy()
        out = out[(out["quali_best_time"] > 0)]
        if out.empty:
            return pd.DataFrame()

        # 6) Derivadas (ratio/delta solo si hay FP3)
        out["ratio_fp3_quali"] = np.nan
        out["delta_fp3_quali_s"] = np.nan
        mask_fp3 = pd.to_numeric(out["fp3_best_time"], errors="coerce").notna() & (out["fp3_best_time"] > 0)
        out.loc[mask_fp3, "ratio_fp3_quali"] = out.loc[mask_fp3, "quali_best_time"] / out.loc[mask_fp3, "fp3_best_time"]
        out.loc[mask_fp3, "delta_fp3_quali_s"] = out.loc[mask_fp3, "quali_best_time"] - out.loc[mask_fp3, "fp3_best_time"]
        # winsor simple
        rl, rh = 0.05, 0.95
        if out["ratio_fp3_quali"].notna().any():
            out["ratio_fp3_quali_w"] = out["ratio_fp3_quali"].clip(
                lower=out["ratio_fp3_quali"].quantile(rl), upper=out["ratio_fp3_quali"].quantile(rh)
            )
        else:
            out["ratio_fp3_quali_w"] = np.nan
        if out["delta_fp3_quali_s"].notna().any():
            out["delta_fp3_quali_w_s"] = out["delta_fp3_quali_s"].clip(
                lower=out["delta_fp3_quali_s"].quantile(rl), upper=out["delta_fp3_quali_s"].quantile(rh)
            )
        else:
            out["delta_fp3_quali_w_s"] = np.nan
        # proxy por evento
        ev = out.groupby("weekend_key")["delta_fp3_quali_w_s"].median().rename("event_delta_median_s")
        out = out.merge(ev, on="weekend_key", how="left")

        # 7) Guardar CSV
        cache_dir = Path("app/models_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cache_dir / "quali_dataset_latest.csv"
        out_cols = [
            "driver", "team", "race_name", "year", "round", "weekend_key",
            "fp3_best_time", "quali_best_time", "ratio_fp3_quali", "delta_fp3_quali_s",
            "fp3_laps_count", "event_delta_median_s",
            # Sprint-related
            "sq_best_time", "sq_position", "sprint_position", "sprint_best_lap_time", "sprint_points"
        ]
        for c in ["fp3_laps_count"]:
            if c not in out.columns:
                out[c] = np.nan
        # Ensure sprint columns exist
        for c in ["sq_best_time", "sq_position", "sprint_position", "sprint_best_lap_time", "sprint_points"]:
            if c not in out.columns:
                out[c] = np.nan
        out[out_cols].to_csv(csv_path, index=False)
        print(f"💾 Dataset FP3→Quali guardado: {csv_path}")
        return out[out_cols]
