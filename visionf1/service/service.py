"""
Prediction Service for VisionF1.
Handles the orchestration of training and prediction pipelines.
"""
import logging
import os
from threading import Lock
from typing import Dict, Any, List, Optional
import pandas as pd

from visionf1.core.pipeline import Pipeline
from visionf1.config import RACE_RANGE, PREDICTION_CONFIG, SCENARIO_EMOJIS, VALID_RACE_NAMES, VALID_WEATHER_SCENARIOS, RACE_PREDICTION
from visionf1.utils.cache import load_cache, save_cache
from visionf1.models.models import NextRaceInfo

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.pipeline = Pipeline(RACE_RANGE)
        self.cache_lock = Lock()
        
        # Cache directories directory relative to where the app is run (root)
        self.base_cache_dir = "visionf1/models_cache/api_cache"
        self.race_cache_dir = os.path.join(self.base_cache_dir, "predict_race")
        self.quali_cache_dir = os.path.join(self.base_cache_dir, "predict_quali")
        self.all_cache_dir = os.path.join(self.base_cache_dir, "predict_all")
        
        os.makedirs(self.race_cache_dir, exist_ok=True)
        os.makedirs(self.quali_cache_dir, exist_ok=True)
        os.makedirs(self.all_cache_dir, exist_ok=True)

    def get_next_race_info(self, race_name: Optional[str] = None, weather_scenario: Optional[str] = None) -> Dict[str, Any]:
        cfg = PREDICTION_CONFIG
        race_cfg = cfg["next_race"]

        final_race_name = race_name or race_cfg["race_name"]
        final_scenario = weather_scenario or cfg["active_scenario"]
        final_emoji = SCENARIO_EMOJIS.get(final_scenario, '')
        
        return {
            "race_name": final_race_name,
            "season": 2025,
            "active_scenario": final_scenario,
            "active_scenario_emoji": final_emoji,
        }

    def _build_quali_top(self, top=10) -> List[Dict[str, Any]]:
        """Reads CSV predictions and returns top N as JSON."""
        qp = "visionf1/models_cache/quali_predictions_latest.csv"
        try:
            df = pd.read_csv(qp)
            df = df.sort_values("pred_rank").head(top)
            results = []
            for _, r in df.iterrows():
                results.append({
                    "driver": r["driver"],
                    "team": r["team"],
                    "race_name": r.get("race_name"),
                    "pred_rank": int(r["pred_rank"]),
                    "pred_best_quali_lap": r["pred_best_quali_lap"],
                })
            return results
        except Exception as e:
            logger.error(f"Error building quali top: {e}")
            return []

    def _build_race_full(self) -> List[Dict[str, Any]]:
        """Reads CSV race predictions and returns all."""
        rp = "visionf1/models_cache/race_predictions_latest.csv"
        try:
            df = pd.read_csv(rp)
            df = df.sort_values("final_position")
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "driver": r["driver"],
                    "team": r["team"],
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
            cached = load_cache(self.race_cache_dir, *key_race)

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
            results = self._build_race_full()

            response = {
                "status": "ok",
                "detail": "Race position prediction generated",
                "cached": False,
                "next_race": info,
                "race_predictions": results,
            }

            with self.cache_lock:
                save_cache(self.race_cache_dir, *key_race, {k: v for k, v in response.items() if k != "cached"})
            
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
            cached = load_cache(self.quali_cache_dir, *key_quali)

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
                save_cache(self.quali_cache_dir, *key_quali, {k: v for k, v in response.items() if k != "cached"})
            
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
             cached = load_cache(self.all_cache_dir, *key_all)
        
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
                race_full = self._build_race_full()
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
                save_cache(self.all_cache_dir, *key_all, response_core)
            
            return response
         finally:
            cfg["next_race"] = original_next_race
            cfg["active_scenario"] = original_active_scenario
            cfg["active_scenario_emoji"] = original_active_emoji
