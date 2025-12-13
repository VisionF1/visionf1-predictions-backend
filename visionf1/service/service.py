"""
Prediction Service for VisionF1.
Handles the orchestration of training and prediction pipelines.
"""
import os
import json
import logging
from typing import Any, Dict, List, Optional
from threading import Lock
import pandas as pd

import unicodedata
import re

from app.core.pipeline import Pipeline
from app.config import RACE_RANGE, PREDICTION_CONFIG, SCENARIO_EMOJIS
from visionf1.config import TEAM_DISPLAY_MAPPING

# Cache Constants
BASE_CACHE_DIR = "app/models_cache/api_cache"
RACE_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "predict_race")
QUALI_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "predict_quali")
ALL_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "predict_all")

logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(RACE_CACHE_DIR, exist_ok=True)
os.makedirs(QUALI_CACHE_DIR, exist_ok=True)
os.makedirs(ALL_CACHE_DIR, exist_ok=True)


class PredictionService:
    def __init__(self):
        # Initialize pipeline once
        self.pipeline = Pipeline(RACE_RANGE)
        self.cache_lock = Lock()

    def _slugify(self, text: str) -> str:
        norm = unicodedata.normalize("NFKD", text)
        norm = norm.encode("ascii", "ignore").decode("ascii")
        norm = norm.lower()
        norm = re.sub(r"[^a-z0-9]+", "_", norm)
        norm = norm.strip("_")
        return norm or "unknown"

    def _cache_path(self, base_dir: str, race_name: str, scenario: str) -> str:
        race_slug = self._slugify(race_name)
        scen_slug = self._slugify(scenario)
        filename = f"{race_slug}__{scen_slug}.json"
        return os.path.join(base_dir, filename)

    def _load_cache(self, base_dir: str, race_name: str, scenario: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(base_dir, race_name, scenario)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cache(self, base_dir: str, race_name: str, scenario: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(base_dir, race_name, scenario)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            pass

    def get_next_race_info(self, race_name: str | None = None, weather_scenario: str | None = None) -> Dict[str, Any]:
        cfg = PREDICTION_CONFIG
        race_cfg = cfg.get("next_race", {})

        final_race_name = race_name or race_cfg.get("race_name")
        final_scenario = weather_scenario or cfg.get("active_scenario")
        final_emoji = SCENARIO_EMOJIS.get(final_scenario, '')
        
        return {
            "race_name": final_race_name,
            "season": 2025,
            "active_scenario": final_scenario,
            "active_scenario_emoji": final_emoji,
        }

    def _build_quali_top(self, top=10) -> List[Dict[str, Any]]:
        qp = "app/models_cache/quali_predictions_latest.csv"
        try:
            df = pd.read_csv(qp)
            df = df.sort_values("pred_rank").head(top)
            results = []
            for _, r in df.iterrows():
                results.append({
                    "driver": r["driver"],
                    "team": TEAM_DISPLAY_MAPPING.get(r["team"], r["team"]),
                    "race_name": r.get("race_name"),
                    "pred_rank": int(r["pred_rank"]),
                    "pred_best_quali_lap": r["pred_best_quali_lap"],
                })
            return results
        except Exception as e:
            logger.error(f"Error building quali top: {e}")
            return []

    def _build_race_full(self, filename: str = "race_predictions_latest.csv") -> List[Dict[str, Any]]:
        """Reads CSV race predictions and returns all."""
        rp = f"app/models_cache/{filename}"
        try:
            df = pd.read_csv(rp)
            df = df.sort_values("final_position")
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "driver": r["driver"],
                    "team": TEAM_DISPLAY_MAPPING.get(r["team"], r["team"]),
                    "final_position": int(r["final_position"]),
                })
            return rows
        except Exception as e:
            logger.error(f"Error building race full: {e}")
            return []

    def predict_race(self, race_name: Optional[str], scenario: Optional[str]) -> Dict[str, Any]:
        cfg = PREDICTION_CONFIG
        current_next_race = cfg["next_race"]
        current_scenario = cfg["active_scenario"]

        race_name = race_name or current_next_race["race_name"]
        scenario = scenario or current_scenario

        key_race = (race_name, scenario)

        with self.cache_lock:
            cached = self._load_cache(RACE_CACHE_DIR, *key_race)
        
        if cached is not None:
            logger.info(f"Using disk cache for {key_race}")
            return {**cached, "cached": True}

        # No cache, run logic
        original_next_race = current_next_race.copy()
        original_active_scenario = cfg["active_scenario"]
        original_active_emoji = cfg.get("active_scenario_emoji")

        try:
            cfg["next_race"]["race_name"] = race_name
            cfg["active_scenario"] = scenario
            
            info = self.get_next_race_info(race_name, scenario)
            self.pipeline.predict_next_race_positions()
            results = self._build_race_full(filename="realistic_predictions_2025.csv")

            response = {
                "status": "ok",
                "detail": "Race position prediction generated",
                "cached": False,
                "next_race": info,
                "race_predictions": results,
            }

            with self.cache_lock:
                self._save_cache(RACE_CACHE_DIR, *key_race, {k: v for k, v in response.items() if k != "cached"})
            
            return response
        finally:
            cfg["next_race"] = original_next_race
            cfg["active_scenario"] = original_active_scenario
            cfg["active_scenario_emoji"] = original_active_emoji

    def predict_quali(self, race_name: Optional[str], scenario: Optional[str]) -> Dict[str, Any]:
        cfg = PREDICTION_CONFIG
        current_next_race = cfg["next_race"]
        current_scenario = cfg["active_scenario"]

        race_name = race_name or current_next_race["race_name"]
        scenario = scenario or current_scenario

        key_quali = (race_name, scenario)

        with self.cache_lock:
            cached = self._load_cache(QUALI_CACHE_DIR, *key_quali)
            
        if cached is not None:
             logger.info(f"Using disk cache for {key_quali}")
             return {**cached, "cached": True}
        
        original_next_race = current_next_race.copy()
        original_active_scenario = cfg["active_scenario"]
        original_active_emoji = cfg.get("active_scenario_emoji")

        try:
            cfg["next_race"]["race_name"] = race_name
            cfg["active_scenario"] = scenario

            info = self.get_next_race_info(race_name, scenario)
            self.pipeline.predict_quali_next_race()
            quali_top = self._build_quali_top(top=20)

            response = {
                "status": "ok",
                "detail": "Quali prediction generated",
                "cached": False,
                "next_race": info,
                "quali_predicts": quali_top,
            }

            with self.cache_lock:
                self._save_cache(QUALI_CACHE_DIR, *key_quali, {k: v for k, v in response.items() if k != "cached"})
            
            return response
        finally:
             cfg["next_race"] = original_next_race
             cfg["active_scenario"] = original_active_scenario
             cfg["active_scenario_emoji"] = original_active_emoji

    def predict_all(self, race_name: Optional[str], scenario: Optional[str]) -> Dict[str, Any]:
         cfg = PREDICTION_CONFIG
         current_next_race = cfg["next_race"]
         current_scenario = cfg["active_scenario"]

         race_name = race_name or current_next_race["race_name"]
         scenario = scenario or current_scenario

         key_all = (race_name, scenario)

         with self.cache_lock:
             cached = self._load_cache(ALL_CACHE_DIR, *key_all)
        
         if cached is not None:
             logger.info(f"Using disk cache for {key_all}")
             return {**cached, "cached": True}

         original_next_race = current_next_race.copy()
         original_active_scenario = cfg["active_scenario"]
         original_active_emoji = cfg.get("active_scenario_emoji")

         try:
            cfg["next_race"]["race_name"] = race_name
            cfg["active_scenario"] = scenario

            info = self.get_next_race_info(race_name, scenario)
            self.pipeline.predict_all() # This creates artifacts

            race_full = []
            quali_top10 = []
            errors = {}

            try:
                race_full = self._build_race_full(filename="race_predictions_latest.csv")
            except Exception as e:
                errors["race_full"] = str(e)
            
            try:
                quali_top10 = self._build_quali_top(top=10)
            except Exception as e:
                errors["quali_top10"] = str(e)
            
            response_core = {
                "status": "ok" if not errors else "partial_ok",
                "detail": "Quali + Race predictions executed",
                "next_race": info,
                "quali_top10": quali_top10,
                "race_predictions_full": race_full,
                "errors": errors or None
            }

            response = {**response_core, "cached": False}

            with self.cache_lock:
                self._save_cache(ALL_CACHE_DIR, *key_all, response_core)
            
            return response
         finally:
            cfg["next_race"] = original_next_race
            cfg["active_scenario"] = original_active_scenario
            cfg["active_scenario_emoji"] = original_active_emoji
