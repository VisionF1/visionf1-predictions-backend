"""
Prediction Controller.
Handles HTTP requests for predictions.
"""
from fastapi import HTTPException

from visionf1.service.service import PredictionService
from visionf1.models.models import PredictParams, NextRaceInfo, PredictRaceResponse, PredictQualiResponse, PredictAllResponse
from visionf1.config import VALID_RACE_NAMES, VALID_WEATHER_SCENARIOS

class PredictionController:
    def __init__(self):
        self.service = PredictionService()

    def get_next_race_info(self) -> NextRaceInfo:
        return self.service.get_next_race_info()

    def _validate_params(self, params: PredictParams):
        if params.race_name and params.race_name not in VALID_RACE_NAMES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_race_name",
                    "message": "race_name must be a valid race name",
                    "valid_race_names": VALID_RACE_NAMES,
                },
            )

        if params.weather_scenario and params.weather_scenario not in VALID_WEATHER_SCENARIOS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_weather_scenario",
                    "message": "weather_scenario is invalid",
                    "valid_weather_scenarios": VALID_WEATHER_SCENARIOS,
                },
            )

    def predict_race(self, params: PredictParams) -> PredictRaceResponse:
        self._validate_params(params)
        return self.service.predict_race(params.race_name, params.weather_scenario)

    def predict_quali(self, params: PredictParams) -> PredictQualiResponse:
        self._validate_params(params)
        return self.service.predict_quali(params.race_name, params.weather_scenario)

    def predict_all(self, params: PredictParams) -> PredictAllResponse:
        self._validate_params(params)
        return self.service.predict_all(params.race_name, params.weather_scenario)

# Global instance
prediction_controller = PredictionController()
