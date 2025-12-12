"""
Advanced Feature Engineering (pre-race safe)
-------------------------------------------
Genera únicamente características disponibles ANTES del fin de semana.
No usa libres, quali ni información post-carrera.

Incluye:
- Normalización de nombres de GP y cálculo de 'round' por año usando fastf1.
- Compatibilidad de circuito (street/power/hybrid).
- Momentum de piloto y equipo basado en histórico previo

Requisitos mínimos de entrada:
- 'year', 'race_name', 'driver', 'team' (recomendado), 'points' (para momentum).
"""

from __future__ import annotations
import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings("ignore")


# -------------------- Helpers de calendario / nombres --------------------

# Sinónimos comunes para nombres de GP -> nombre canónico
RACE_SYNONYMS: Dict[str, str] = {
    # Brasil
    "Brazilian Grand Prix": "São Paulo Grand Prix",
    "Sao Paulo Grand Prix": "São Paulo Grand Prix",
    # Estados Unidos
    "United States GP": "United States Grand Prix",
    "United States Grand Prix (Austin)": "United States Grand Prix",
    # México
    "Mexican Grand Prix": "Mexico City Grand Prix",
    # Italia (Imola/Monza)
    "Emilia-Romagna Grand Prix": "Emilia Romagna Grand Prix",
    # Países Bajos
    "Netherlands Grand Prix": "Dutch Grand Prix",
    # EAU
    "Abu Dhabi GP": "Abu Dhabi Grand Prix",
    # Otros alias frecuentes
    "Qatar GP": "Qatar Grand Prix",
    "Chinese GP": "Chinese Grand Prix",
    "Japan GP": "Japanese Grand Prix",
}

def normalize_gp_name(name: str) -> str:
    """Lleva el nombre de carrera a un canónico estable."""
    if not isinstance(name, str):
        return name
    s = name.strip()
    return RACE_SYNONYMS.get(s, s)


def build_year_round_map(years: List[int]) -> Dict[int, Dict[str, int]]:
    """
    Construye un mapa {year: {EventNameCanonico: RoundNumber}} usando fastf1.
    Solo consulta los años provistos (evita overhead).
    """
    year_round_map: Dict[int, Dict[str, int]] = {}
    for y in years:
        try:
            sched = fastf1.get_event_schedule(int(y))
            mapping: Dict[str, int] = {}
            for _, row in sched.iterrows():
                event_name = str(row.get("EventName", "")).strip()
                if not event_name:
                    continue
                canon = normalize_gp_name(event_name)
                rnd = row.get("RoundNumber")
                if pd.notna(rnd):
                    mapping[canon] = int(rnd)
            year_round_map[int(y)] = mapping
        except Exception:
            # Si falla por algún año, lo dejamos vacío
            year_round_map[int(y)] = {}
    return year_round_map


def add_round_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columna 'race_name_norm' y 'round' al DataFrame a partir del calendario.
    No pisa valores existentes si ya están.
    """
    df = df.copy()
    if "year" not in df.columns or "race_name" not in df.columns:
        return df

    # Normalizar nombres
    if "race_name_norm" not in df.columns:
        df["race_name_norm"] = df["race_name"].apply(normalize_gp_name)

    # Si ya hay 'round' y no es todo NaN, respetamos
    if "round" in df.columns and df["round"].notna().any():
        return df

    years_present = (
        pd.Series(df["year"].unique())
        .dropna()
        .astype(int)
        .tolist()
    )
    ymap = build_year_round_map(years_present)

    def _map_round(row):
        y = int(row["year"])
        name = row["race_name_norm"]
        return ymap.get(y, {}).get(name, np.nan)

    df["round"] = df.apply(_map_round, axis=1).astype("Float64")
    return df


# -------------------- Ingeniero de Features --------------------

class AdvancedFeatureEngineer:
    """
    Ingeniero de features minimalista, 100% pre-race-safe.

    Métodos expuestos:
      - create_circuit_compatibility_features(df)
      - create_momentum_features(df)
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.created_features: List[str] = []

        # Mapeo básico GP -> tipo de circuito (ajustable)
        self._circuit_type_map: Dict[str, str] = {
            # Callejeros / urbanos
            "Monaco Grand Prix": "street",
            "Azerbaijan Grand Prix": "street",
            "Singapore Grand Prix": "street",
            "Las Vegas Grand Prix": "street",
            "Miami Grand Prix": "street",
            "Saudi Arabian Grand Prix": "street",
            "Australian Grand Prix": "street",  # Albert Park semi-urbano

            # Power tracks
            "Italian Grand Prix": "power",      # Monza
            "Belgian Grand Prix": "power",      # Spa
            "British Grand Prix": "power",      # Silverstone (mixto pero aero/power)
            "Canadian Grand Prix": "power",     # Rectas + chicanas

            # Híbridos (default)
            "Bahrain Grand Prix": "hybrid",
            "Spanish Grand Prix": "hybrid",
            "Hungarian Grand Prix": "hybrid",
            "Austrian Grand Prix": "hybrid",
            "Emilia Romagna Grand Prix": "hybrid",
            "French Grand Prix": "hybrid",
            "Portuguese Grand Prix": "hybrid",
            "Turkish Grand Prix": "hybrid",
            "United States Grand Prix": "hybrid",
            "Mexico City Grand Prix": "hybrid",
            "São Paulo Grand Prix": "hybrid",
            "Japanese Grand Prix": "hybrid",
            "Qatar Grand Prix": "hybrid",
            "Abu Dhabi Grand Prix": "hybrid",
            "Dutch Grand Prix": "hybrid",
        }

    # -------------------- utils --------------------

    def _log(self, msg: str):
        if not self.quiet or os.getenv("VISIONF1_DEBUG", "0") == "1":
            print(msg)

    @staticmethod
    def _safe_num(s: pd.Series, fill: float = 0.0) -> pd.Series:
        return pd.to_numeric(s, errors="coerce").fillna(fill)

    @staticmethod
    def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    # -------------------- features pre-race --------------------

    def create_circuit_compatibility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea:
          - race_name_norm (string canónico)
          - circuit_type (string)
          - circuit_type_encoded (0/1/2)
          - is_street, is_power, is_hybrid (dummies numéricas)
        """
        df = df.copy()
        df = self._ensure_columns(df, ["race_name"])

        # Normalizar nombre de GP
        df["race_name_norm"] = df["race_name"].apply(normalize_gp_name)

        # Mapear a tipo
        def map_type(name: Optional[str]) -> str:
            if not isinstance(name, str):
                return "hybrid"
            return self._circuit_type_map.get(name, "hybrid")

        df["circuit_type"] = df["race_name_norm"].apply(map_type)

        # Codificación simple y dummies
        type_to_int = {"street": 0, "power": 1, "hybrid": 2}
        df["circuit_type_encoded"] = df["circuit_type"].map(type_to_int).fillna(2).astype(int)

        df["is_street"] = (df["circuit_type"] == "street").astype(int)
        df["is_power"] = (df["circuit_type"] == "power").astype(int)
        df["is_hybrid"] = (df["circuit_type"] == "hybrid").astype(int)

        self.created_features += [
            "race_name_norm",
            "circuit_type",
            "circuit_type_encoded",
            "is_street",
            "is_power",
            "is_hybrid",
        ]
        self._log("✅ Features de compatibilidad de circuito creadas (pre-race safe).")
        return df

    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construye momentum PRE-RACE usando únicamente histórico anterior.
          - Añade 'round' (si falta) en base a calendario fastf1.
          - Ordena por ['year','round'] si está disponible (sino fallback).
          - Usa shift(1) para excluir SIEMPRE la carrera actual.

        Genera:
          - driver_points_last_3 / last_5 (rolling mean previo)
          - driver_points_cumavg (promedio acumulado previo)
          - driver_races_count_prior
          - driver_competitiveness (EMA de puntos previos)
          - team_competitiveness (EMA de puntos del equipo por carrera)
          - points_last_3 (alias)
        """
        df = df.copy()
        df = self._ensure_columns(df, ["driver", "team", "year", "race_name", "points"])

        # Asegurar 'race_name_norm' y 'round'
        df = add_round_column(df)

        # Orden temporal
        order_cols = ["year"]
        if "round" in df.columns and df["round"].notna().any():
            df["round"] = pd.to_numeric(df["round"], errors="coerce")
            order_cols.append("round")
        else:
            df["_row_ix"] = np.arange(len(df))
            order_cols.append("_row_ix")

        df = df.sort_values(order_cols).reset_index(drop=True)

        # Normalizar puntos
        df["points"] = self._safe_num(df["points"], fill=0.0)

        # ---- Momentum por piloto ----
        def _driver_rolling(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            prior_points = g["points"]
            g["driver_points_last_3"] = prior_points.rolling(3, min_periods=1).mean()
            g["driver_points_last_5"] = prior_points.rolling(5, min_periods=1).mean()
            g["driver_points_cumavg"] = prior_points.expanding(min_periods=1).mean()
            g["driver_races_count_prior"] = (~prior_points.isna()).astype(int).cumsum()
            g["driver_competitiveness"] = prior_points.ewm(alpha=0.4, adjust=False, min_periods=1).mean()
            return g

        df = df.groupby("driver", group_keys=False).apply(_driver_rolling)

        # Alias esperado
        if "points_last_3" not in df.columns:
            df["points_last_3"] = df["driver_points_last_3"]

        # ---- Momentum por equipo (EMA de puntos agregados por carrera) ----
        def _team_ema(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            # Clave de carrera para agregar puntos del equipo por evento
            race_key = ["year"]
            if "race_name_norm" in g.columns:
                race_key.append("race_name_norm")
            elif "round" in g.columns:
                race_key.append("round")

            tmp = (
                g.groupby(race_key)["points"]
                .sum()
                .reset_index()
                .sort_values(race_key)
                .reset_index(drop=True)
            )
            tmp["team_points_prior"] = tmp["points"]
            tmp["team_competitiveness_teamlvl"] = (
                tmp["team_points_prior"].ewm(alpha=0.4, adjust=False, min_periods=1).mean()
            )

            g = g.merge(tmp[race_key + ["team_competitiveness_teamlvl"]], on=race_key, how="left")
            return g

        df = df.groupby("team", group_keys=False).apply(_team_ema)
        # Relleno estable
        fill_val = df["team_competitiveness_teamlvl"].median(skipna=True)
        df["team_competitiveness"] = df["team_competitiveness_teamlvl"].fillna(fill_val if pd.notna(fill_val) else 0.0)
        df.drop(columns=["team_competitiveness_teamlvl"], errors="ignore", inplace=True)

        # Saneos finales
        fill_map = {
            "driver_points_last_3": 0.0,
            "driver_points_last_5": 0.0,
            "driver_points_cumavg": 0.0,
            "driver_races_count_prior": 0,
            "driver_competitiveness": 0.0,
            "team_competitiveness": 0.0,
            "points_last_3": 0.0,
        }
        for c, v in fill_map.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)

        # Limpiar auxiliar si existe
        if "_row_ix" in df.columns:
            df.drop(columns=["_row_ix"], inplace=True, errors="ignore")

        self.created_features += [
            "driver_points_last_3",
            "driver_points_last_5",
            "driver_points_cumavg",
            "driver_races_count_prior",
            "driver_competitiveness",
            "team_competitiveness",
            "points_last_3",
            "round",
            "race_name_norm",
        ]
        self._log("✅ Features de momentum creadas (pre-race, excluye carrera actual).")
        return df
    

    def create_weather_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rendimiento histórico condicionado por clima (lluvia vs seco), 100% pre-race safe.
        Crea:
        - driver_avg_points_in_rain
        - driver_avg_points_in_dry
        - driver_rain_dry_delta
        """
        df = df.copy()
        df = self._ensure_columns(
            df, ["driver", "year", "race_name", "points", "session_rainfall"]
        )

        # Round + orden temporal
        df = add_round_column(df)
        order_cols = ["year"]
        if "round" in df.columns and df["round"].notna().any():
            df["round"] = pd.to_numeric(df["round"], errors="coerce")
            order_cols.append("round")
        else:
            df["_row_ix_wthr"] = np.arange(len(df))
            order_cols.append("_row_ix_wthr")
        df = df.sort_values(order_cols).reset_index(drop=True)

        # Normalizaciones
        df["points"] = self._safe_num(df["points"], fill=0.0)
        # lluvia -> {0,1} robusto
        if df["session_rainfall"].dtype != bool:
            df["session_rainfall"] = (
                df["session_rainfall"].astype(str).str.strip().str.lower()
                .map({"true":1, "1":1, "yes":1, "false":0, "0":0, "no":0})
                .fillna(0).astype(int)
            )
        else:
            df["session_rainfall"] = df["session_rainfall"].astype(int)

        def _by_driver(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            pts  = g["points"].astype(float)
            rain = g["session_rainfall"].astype(int)
            dry  = 1 - rain

            rain_pts_cumsum = (pts * rain).cumsum()
            rain_cnt_cumsum = rain.cumsum()
            dry_pts_cumsum  = (pts * dry).cumsum()
            dry_cnt_cumsum  = dry.cumsum()

            g["driver_avg_points_in_rain"] = np.where(
                rain_cnt_cumsum > 0, rain_pts_cumsum / rain_cnt_cumsum, 0.0
            )
            g["driver_avg_points_in_dry"] = np.where(
                dry_cnt_cumsum > 0, dry_pts_cumsum / dry_cnt_cumsum, 0.0
            )
            g["driver_rain_dry_delta"] = (
                g["driver_avg_points_in_rain"] - g["driver_avg_points_in_dry"]
            )
            return g

        df = df.groupby("driver", group_keys=False).apply(_by_driver)

        if "_row_ix_wthr" in df.columns:
            df.drop(columns=["_row_ix_wthr"], inplace=True, errors="ignore")

        self.created_features += [
            "driver_avg_points_in_rain",
            "driver_avg_points_in_dry",
            "driver_rain_dry_delta",
        ]
        self._log("✅ Features climáticas creadas (robusto).")
        return df

    # -------------------- utilidades de inspección --------------------

    def list_created_features(self) -> List[str]:
        """Lista de features creadas en esta instancia."""
        return sorted(set(self.created_features))
