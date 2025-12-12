import os
import pickle
import numpy as np
import pandas as pd
from app.core.training.enhanced_data_preparer import EnhancedDataPreparer
from .paths import BEFORE_FE_PATH, CACHED_DATA_PKL

class InferenceDatasetBuilder:
    def __init__(self, logger, feature_helper, manifest_helper):
        self._log = logger.log
        self.dp = EnhancedDataPreparer(quiet=True)
        self.fh = feature_helper
        self.mh = manifest_helper
        self._last_weather_stats = None

    @property
    def last_weather_stats(self):
        return self._last_weather_stats

    def _ensure_hist(self):
        hist_df = None
        if os.path.exists(CACHED_DATA_PKL):
            try:
                hist_df = pickle.load(open(CACHED_DATA_PKL, "rb"))
                if not isinstance(hist_df, pd.DataFrame):
                    hist_df = pd.DataFrame(hist_df)
            except Exception as e:
                self._log(f"⚠️ No se pudo cargar histórico (PKL): {e}")
        if (hist_df is None or hist_df.empty) and os.path.exists(BEFORE_FE_PATH):
            try:
                hist_df = pd.read_csv(BEFORE_FE_PATH)
            except Exception as e:
                self._log(f"⚠️ No se pudo leer histórico (CSV): {e}")
        return hist_df

    def build_X(self, base_df: pd.DataFrame) -> pd.DataFrame:
        year = int(base_df["year"].iloc[0])
        race_name = str(base_df["race_name"].iloc[0])
        hist_df = self._ensure_hist()

        if hist_df is None or hist_df.empty:
            self._log("ℹ️ Sin histórico: genero features SOLO con base_df")
            X_only, _, _, _ = self.dp.prepare_enhanced_features(base_df.copy(), inference=True)
            X_only = pd.DataFrame(X_only) if not isinstance(X_only, pd.DataFrame) else X_only
            if X_only.shape[0] == 0:
                raise ValueError("prepare_enhanced_features(base_df) devolvió 0 filas.")
            if X_only.shape[0] != base_df.shape[0]:
                self._log(f"⚠️ Cardinalidad inesperada: X_only={X_only.shape[0]} vs base={base_df.shape[0]}. Ajusto.")
                X_only = X_only.iloc[: base_df.shape[0]].copy()
            X_only.index = base_df.index
            return self.fh.align_columns(X_only)

        for c in ["driver", "team", "year", "race_name", "points"]:
            if c not in hist_df.columns:
                hist_df[c] = np.nan
        for c, val in [("session_air_temp", 25.0),
                       ("session_track_temp", 35.0),
                       ("session_humidity", 60.0),
                       ("session_rainfall", 0)]:
            if c not in hist_df.columns:
                hist_df[c] = val

        base_marked = base_df[["driver","team","year","race_name",
                               "session_air_temp","session_track_temp","session_humidity","session_rainfall"]].copy()
        base_marked["points"] = np.nan
        base_marked["is_current"] = 1

        hist_marked = hist_df[["driver","team","year","race_name","points",
                               "session_air_temp","session_track_temp","session_humidity","session_rainfall"]].copy()
        hist_marked["is_current"] = 0

        combo_df = pd.concat([hist_marked, base_marked], ignore_index=True)
        combo_df = self.fh.add_weather_perf_features(combo_df)

        try:
            hist_only = combo_df[combo_df["is_current"] == 0].copy()
            order_cols = ["year"]
            if "round" in hist_only.columns and hist_only["round"].notna().any():
                hist_only["round"] = pd.to_numeric(hist_only["round"], errors="coerce")
                order_cols.append("round")
            else:
                hist_only["_row_ix"] = np.arange(len(hist_only))
                order_cols.append("_row_ix")
            hist_only = hist_only.sort_values(order_cols)
            stats = (
                hist_only.groupby("driver", as_index=False).last()[[
                    "driver",
                    "driver_avg_points_in_rain",
                    "driver_avg_points_in_dry",
                    "driver_rain_dry_delta",
                ]]
            )
            self._last_weather_stats = stats.reset_index(drop=True)
        except Exception:
            self._last_weather_stats = None

        X_all, _, _, _ = self.dp.prepare_enhanced_features(combo_df.copy(), inference=True)
        X_all = pd.DataFrame(X_all) if not isinstance(X_all, pd.DataFrame) else X_all

        if "driver" not in X_all.columns:
            if "driver_encoded" in X_all.columns:
                try:
                    dec = self.mh.decode_series("driver", X_all["driver_encoded"])
                    X_all = X_all.copy()
                    X_all.insert(0, "driver", dec)
                except Exception:
                    X_all.insert(0, "driver", base_df.index.tolist())
            else:
                X_all.insert(0, "driver", base_df.index.tolist())

        if self._last_weather_stats is not None and not self._last_weather_stats.empty:
            stats = self._last_weather_stats.copy()
            X_all = X_all.merge(stats, on="driver", how="left")
        else:
            for c in ["driver_avg_points_in_rain", "driver_avg_points_in_dry", "driver_rain_dry_delta"]:
                if c not in X_all.columns:
                    X_all[c] = 0.0

        cur = pd.DataFrame(); used_hybrid = False
        race_code = None
        if self.mh.manifest:
            try:
                race_code = self.mh.encode_value("race_name", race_name)
            except Exception:
                race_code = None

        has_year = "year" in X_all.columns
        has_rne  = "race_name_encoded" in X_all.columns
        has_dre  = "driver_encoded" in X_all.columns

        def _base_driver_codes():
            codes = []
            for d in base_df.index.tolist():
                enc_val = self.mh.encode_value("driver", d)
                if enc_val is not None:
                    codes.append(enc_val)
            return set(codes)

        if (has_year and has_rne and has_dre) and (race_code is not None):
            base_codes = _base_driver_codes()
            cur_try = X_all[(X_all["year"] == year) & (X_all["race_name_encoded"] == race_code)].copy()
            if base_codes:
                cur_try = cur_try[cur_try["driver_encoded"].isin(base_codes)].copy()
            if len(cur_try) > len(base_df):
                cur_try = (cur_try.reset_index(drop=True).drop_duplicates(subset=["driver_encoded"], keep="last"))
                self._log(f"🔎 depurado a última por driver_encoded -> filas={len(cur_try)}")
            if len(cur_try) == len(base_df):
                cur = cur_try.copy()

        if cur.empty and has_dre:
            base_codes = _base_driver_codes()
            last_per_driver = (X_all.reset_index(drop=True).drop_duplicates(subset=["driver_encoded"], keep="last"))
            cur_try = last_per_driver.copy()
            if base_codes:
                cur_try = cur_try[cur_try["driver_encoded"].isin(base_codes)].copy()
            if (len(cur_try) == 0) and ("driver_encoded" in last_per_driver.columns):
                try:
                    dec = self.mh.decode_series("driver", last_per_driver["driver_encoded"])
                    last_per_driver = last_per_driver.copy()
                    last_per_driver["driver"] = dec
                    cur_try = last_per_driver[last_per_driver["driver"].isin(base_df.index)].copy()
                    self._log(f"🔎 fallback decode driver → filtro por base_df -> filas={len(cur_try)}")
                except Exception as e:
                    self._log(f"ℹ️ No pude decodificar driver_encoded para fallback: {e}")
            missing = []
            if "driver" in cur_try.columns:
                have_set = set(cur_try["driver"].dropna().tolist())
            elif "driver_encoded" in cur_try.columns:
                try:
                    dec2 = self.mh.decode_series("driver", cur_try["driver_encoded"])
                    have_set = set(dec2.dropna().tolist())
                    cur_try = cur_try.copy(); cur_try.insert(0, "driver", dec2)
                except Exception:
                    have_set = set()
            else:
                have_set = set()
            for d in base_df.index.tolist():
                if d not in have_set:
                    missing.append(d)
            if missing:
                X_only, _, _, _ = self.dp.prepare_enhanced_features(base_df.loc[missing].copy(), inference=True)
                X_only = pd.DataFrame(X_only) if not isinstance(X_only, pd.DataFrame) else X_only
                if "driver" not in X_only.columns:
                    X_only.insert(0, "driver", missing)
                cur = pd.concat([cur_try, X_only], ignore_index=True)
                used_hybrid = True
            else:
                cur = cur_try.copy()

        if cur.empty and "is_current" in X_all.columns:
            cur_try = X_all[X_all["is_current"] == 1].copy()
            self._log(f"🔎 por is_current -> filas={len(cur_try)}")
            if len(cur_try) == len(base_df):
                cur = cur_try.copy()

        if cur.empty:
            self._log("⚠️ No pude aislar actuales; uso SOLO base_df")
            X_only, _, _, _ = self.dp.prepare_enhanced_features(base_df.copy(), inference=True)
            cur = pd.DataFrame(X_only) if not isinstance(X_only, pd.DataFrame) else X_only
            if "driver" not in cur.columns:
                cur.insert(0, "driver", base_df.index.tolist())

        if "driver" not in cur.columns and "driver_encoded" in cur.columns:
            try:
                dec = self.mh.decode_series("driver", cur["driver_encoded"])
                cur.insert(0, "driver", dec)
            except Exception:
                cur.insert(0, "driver", [None]*len(cur))

        if "driver" in cur.columns:
            cur = cur.copy()
            cur = cur.dropna(subset=["driver"])
            cur = cur[~cur["driver"].duplicated(keep="last")]
            cur = cur.set_index("driver", drop=True)
            if cur.index.has_duplicates:
                cur = cur[~cur.index.duplicated(keep="last")]
        else:
            if cur.shape[0] != base_df.shape[0]:
                cur = cur.iloc[: base_df.shape[0]].copy()
            cur.index = base_df.index

        if cur.shape[0] != base_df.shape[0] or not cur.index.equals(base_df.index):
            self._log(f"🔧 Normalizo cardinalidad/orden (cur={cur.shape[0]} vs base={base_df.shape[0]})")
            cur = cur.reindex(base_df.index)

        cur = self.fh.align_columns(cur)
        self._log(f"📦 FE history-augmented{' (híbrido)' if used_hybrid else ''} → filas={len(cur)}, cols={cur.shape[1]}")
        return cur
