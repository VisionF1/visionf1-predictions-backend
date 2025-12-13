import os
import json
import pickle
import pandas as pd
from .paths import FEATURE_NAMES_PKL, MANIFEST_PATHS

class ManifestHelper:
    def __init__(self, logger):
        self._log = logger.log
        self.manifest = self._load_inference_manifest()
        self._enc_cache = {}

    # -------------------- manifiesto --------------------
    def _load_inference_manifest(self):
        for p in MANIFEST_PATHS:
            try:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    return m
            except Exception:
                continue
        return None

    def manifest_dir(self) -> str:
        path = None
        for p in MANIFEST_PATHS:
            if os.path.exists(p):
                path = p
                break
        return os.path.dirname(path) if path else "app/models_cache"

    def feature_names_from_manifest(self) -> list:
        if not self.manifest:
            return []
        names = self.manifest.get("feature_names") or []
        if isinstance(names, (list, tuple)):
            return [str(x) for x in names]
        return []

    # -------------------- encoders --------------------
    def _get_encoder_spec(self, key: str):
        if not self.manifest:
            return None
        enc = self.manifest.get("encoders", {})
        return enc.get(key)

    def resolve_encoder(self, key: str):
        if key in self._enc_cache:
            return self._enc_cache[key]
        spec = self._get_encoder_spec(key)
        enc_obj = None
        if isinstance(spec, (dict, list, tuple)):
            enc_obj = spec
        elif isinstance(spec, str) and spec.endswith(".pkl"):
            path = spec
            if not os.path.exists(path):
                cand = os.path.join(self.manifest_dir(), os.path.basename(spec))
                if os.path.exists(cand):
                    path = cand
            try:
                with open(path, "rb") as f:
                    enc_obj = pickle.load(f)
                self._log(f"📦 Encoder '{key}' cargado (tipo={type(enc_obj).__name__})")
            except Exception as e:
                self._log(f"❌ No pude cargar encoder {key} desde {spec}: {e}")
                enc_obj = None
        else:
            enc_obj = spec
        self._enc_cache[key] = enc_obj
        return enc_obj

    def _norm_helpers(self):
        import re, unicodedata
        def strip_accents(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))
        def norm_up(s: str) -> str:
            return strip_accents(str(s).strip().upper())
        def norm_lo(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", strip_accents(str(s).strip().lower()))
        return norm_up, norm_lo

    def encode_value(self, key: str, raw_value):
        enc = self.resolve_encoder(key)
        if enc is None:
            return None
        norm_up, norm_lo = self._norm_helpers()
        # dict
        if isinstance(enc, dict):
            if raw_value in enc: return enc[raw_value]
            if str(raw_value) in enc: return enc[str(raw_value)]
            nmap = {norm_lo(k): v for k, v in enc.items()}
            return nmap.get(norm_lo(raw_value))
        # list/tuple
        if isinstance(enc, (list, tuple)):
            vals = list(enc)
            for candidate in (raw_value, norm_up(raw_value)):
                if candidate in vals: return vals.index(candidate)
                if str(candidate) in vals: return vals.index(str(candidate))
            nvals = [norm_lo(v) for v in vals]
            nl = norm_lo(raw_value)
            return nvals.index(nl) if nl in nvals else None
        # LabelEncoder
        classes = getattr(enc, "classes_", None)
        if classes is not None:
            vals = [norm_up(v) for v in list(classes)]
            target = norm_up(raw_value)
            if target in vals:
                return vals.index(target)
            return None
        # OrdinalEncoder
        cats = getattr(enc, "categories_", None)
        if cats is not None:
            arr = cats[0] if len(cats) == 1 else cats[0]
            vals = [norm_up(v) for v in list(arr)]
            target = norm_up(raw_value)
            if target in vals:
                return vals.index(target)
            nvals = [norm_lo(v) for v in list(arr)]
            nl = norm_lo(raw_value)
            return nvals.index(nl) if nl in nvals else None
        return None

    def decode_series(self, key: str, code_series: pd.Series) -> pd.Series:
        enc = self.resolve_encoder(key)
        if enc is None:
            return pd.Series([None]*len(code_series), index=code_series.index)
        if isinstance(enc, dict):
            inv = {}
            for k, v in enc.items():
                inv[v] = k; inv[str(v)] = k
            return code_series.map(lambda x: inv.get(x, inv.get(str(x))))
        if isinstance(enc, (list, tuple)):
            vals = list(enc)
            def _idx(x):
                try:
                    i = int(x); return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_idx)
        classes = getattr(enc, "classes_", None)
        if classes is not None:
            vals = [str(x) for x in list(classes)]
            def _le(x):
                try:
                    i = int(x); return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_le)
        cats = getattr(enc, "categories_", None)
        if cats is not None:
            arr = cats[0] if len(cats) == 1 else cats[0]
            vals = [str(x) for x in list(arr)]
            def _oe(x):
                try:
                    i = int(x); return vals[i] if 0 <= i < len(vals) else None
                except Exception:
                    return None
            return code_series.map(_oe)
        return pd.Series([None]*len(code_series), index=code_series.index)

    # -------------------- feature names legacy --------------------
    def load_trained_feature_names(self):
        try:
            if os.path.exists(FEATURE_NAMES_PKL):
                with open(FEATURE_NAMES_PKL, "rb") as f:
                    names = pickle.load(f)
                if isinstance(names, (list, tuple)):
                    return [str(x) for x in names]
                if hasattr(names, "tolist"):
                    return [str(x) for x in names.tolist()]
                return [str(names)]
        except Exception as e:
            self._log(f"⚠️ No se pudieron cargar feature_names entrenadas: {e}")
        return []
