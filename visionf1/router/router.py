"""
API Router configuration.
"""
from fastapi import APIRouter
from visionf1.controller.controller import prediction_controller
from visionf1.models.models import PredictParams, NextRaceInfo, PredictRaceResponse, PredictQualiResponse, PredictAllResponse

router = APIRouter()

@router.get("/", tags=["Health"])
def root():
    """
    Basic health check endpoint.
    """
    return {
        "status": "ok",
        "message": "F1 Prediction API operational",
    }

@router.get("/config-next-race", response_model=NextRaceInfo, tags=["Config"])
def next_race():
    """
    Get information about the default next race.
    """
    return prediction_controller.get_next_race_info()


@router.post("/predict-race", response_model=PredictRaceResponse, tags=["Predictions"])
def predict_race(params: PredictParams):
    """
    Generate race position predictions.
    """
    return prediction_controller.predict_race(params)


@router.post("/predict-quali", response_model=PredictQualiResponse, tags=["Predictions"])
def predict_quali(params: PredictParams):
    """
    Generate qualification predictions.
    """
    return prediction_controller.predict_quali(params)


@router.post("/predict-all", response_model=PredictAllResponse, tags=["Predictions"])
def predict_all(params: PredictParams):
    """
    Generate predictions for both qualification and race.
    """
    return prediction_controller.predict_all(params)
