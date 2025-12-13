"""
API Data models and schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class PredictParams(BaseModel):
    """
    Parameters for prediction requests.
    """
    race_name: Optional[str] = None
    weather_scenario: Optional[str] = None


class NextRaceInfo(BaseModel):
    """
    Information about the next race and scenario.
    """
    race_name: str
    season: int
    active_scenario: str
    active_scenario_emoji: str


class QualiPredictionItem(BaseModel):
    """
    Single qualification prediction item.
    """
    driver: str
    team: str
    race_name: Optional[str] = None
    pred_rank: int
    pred_best_quali_lap: Optional[str] = None


class RacePredictionItem(BaseModel):
    """
    Single race prediction item.
    """
    driver: str
    team: str
    final_position: int


class BasePredictionResponse(BaseModel):
    """
    Base response for prediction endpoints.
    """
    status: str
    detail: str
    cached: bool
    next_race: NextRaceInfo


class PredictRaceResponse(BasePredictionResponse):
    """
    Response for race prediction.
    """
    race_predictions: List[RacePredictionItem]


class PredictQualiResponse(BasePredictionResponse):
    """
    Response for qualification prediction.
    """
    quali_predicts: List[QualiPredictionItem]


class PredictAllResponse(BasePredictionResponse):
    """
    Response for both qualification and race prediction.
    """
    quali_top10: List[QualiPredictionItem]
    race_predictions_full: List[RacePredictionItem]
    errors: Optional[Dict[str, str]] = None
