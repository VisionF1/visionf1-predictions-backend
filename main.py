from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import pandas as pd
import os
import json
import unicodedata
import re
from threading import Lock


from app.core.pipeline import Pipeline
from app.config import RACE_RANGE, PREDICTION_CONFIG, SCENARIO_EMOJIS


VALID_RACE_NAMES = [
    "Abu Dhabi Grand Prix",
    "Australian Grand Prix",
    "Austrian Grand Prix",
    "Azerbaijan Grand Prix",
    "Bahrain Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Canadian Grand Prix",
    "Chinese Grand Prix",
    "Dutch Grand Prix",
    "Emilia Romagna Grand Prix",
    "French Grand Prix",
    "Hungarian Grand Prix",
    "Italian Grand Prix",
    "Japanese Grand Prix",
    "Las Vegas Grand Prix",
    "Mexico City Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Qatar Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Spanish Grand Prix",
    "São Paulo Grand Prix",
    "United States Grand Prix",
]


VALID_WEATHER_SCENARIOS = [
    "dry",
    "hot",
    "wet",
    "storm",
    "cold",
]

app = FastAPI(
    title="F1 Prediction API",
    description="API para entrenar modelos y generar predicciones de F1",
    version="1.0.1",
)

# Instanciamos una sola vez el pipeline al arrancar la app
pipeline = Pipeline(RACE_RANGE)

# Directorios de cache en disco
BASE_CACHE_DIR = "app/models_cache/api_cache"
RACE_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "predict_race")
QUALI_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "predict_quali")
ALL_CACHE_DIR = os.path.join(BASE_CACHE_DIR, "predict_all")

os.makedirs(RACE_CACHE_DIR, exist_ok=True)
os.makedirs(QUALI_CACHE_DIR, exist_ok=True)
os.makedirs(ALL_CACHE_DIR, exist_ok=True)

cache_lock = Lock()


def _slugify(text: str) -> str:
    """Convierte texto en un slug seguro para nombre de archivo."""
    norm = unicodedata.normalize("NFKD", text)
    norm = norm.encode("ascii", "ignore").decode("ascii")
    norm = norm.lower()
    norm = re.sub(r"[^a-z0-9]+", "_", norm)
    norm = norm.strip("_")
    return norm or "unknown"


def _cache_path(base_dir: str, race_name: str, scenario: str) -> str:
    race_slug = _slugify(race_name)
    scen_slug = _slugify(scenario)
    filename = f"{race_slug}__{scen_slug}.json"
    return os.path.join(base_dir, filename)


def load_cache(base_dir: str, race_name: str, scenario: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(base_dir, race_name, scenario)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(base_dir: str, race_name: str, scenario: str, payload: Dict[str, Any]) -> None:
    path = _cache_path(base_dir, race_name, scenario)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass



class PredictParams(BaseModel):
    race_name: Optional[str] = None
    weather_scenario: Optional[str] = None


def get_next_race_info(
    race_name: str | None = None,
    weather_scenario: str | None = None,
) -> Dict[str, Any]:

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

@app.get("/")
def root():
    """
    Endpoint básico de salud.
    Equivalente a un "ping" para ver si la API está arriba.
    """
    return {
        "status": "ok",
        "message": "API de modelo F1 funcionando",
    }


@app.get("/config-next-race")
def next_race():
    """
    Información de la carrera que se simulara por defecto
    """
    return get_next_race_info()


@app.post("/predict-race")
def predict_race(params: PredictParams):
    """
    Predicciones de posiciones para la próxima carrera.

    Body opcional:
    {
      "race_name": "São Paulo Grand Prix",
      "weather_scenario": "dry"
    }
    """
    cfg = PREDICTION_CONFIG

    current_next_race = cfg["next_race"]
    current_scenario = cfg["active_scenario"]

    race_name = params.race_name or current_next_race["race_name"]
    scenario = params.weather_scenario or current_scenario

    # Validaciones
    if params.race_name and race_name not in VALID_RACE_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_race_name",
                "message": "race_name debe estar entre las opciones permitidas",
                "valid_race_names": VALID_RACE_NAMES,
            },
        )

    if params.weather_scenario and scenario not in VALID_WEATHER_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_weather_scenario",
                "message": "weather_scenario inválido",
                "valid_weather_scenarios": VALID_WEATHER_SCENARIOS,
            },
        )

    # Clave de cache
    key_race = (race_name, scenario)

    # Intentar leer de cache en disco
    with cache_lock:
        cached = load_cache(RACE_CACHE_DIR, *key_race)

    if cached is not None:
        print("Usando cache disco para", key_race)
        # Devolvemos lo que hay en disco + flag cached
        return {
            **cached,
            "cached": True,
        }

    # No hay cache: ejecutar pipeline
    original_next_race = current_next_race.copy()
    original_active_scenario = cfg["active_scenario"]
    original_active_emoji = cfg.get("active_scenario_emoji")

    try:
        cfg["next_race"]["race_name"] = race_name
        cfg["active_scenario"] = scenario

        info = get_next_race_info(
            race_name=race_name,
            weather_scenario=scenario,
        )

        pipeline.predict_next_race_positions()
        results = build_race_full()

        response = {
            "status": "ok",
            "detail": "Predicción de posiciones de carrera generada",
            "cached": False,
            "next_race": info,
            "race_predictions": results,
        }

        # Guardar en disco (sin problema si falla)
        with cache_lock:
            save_cache(RACE_CACHE_DIR, *key_race, {k: v for k, v in response.items() if k != "cached"})

        return response

    finally:
        cfg["next_race"] = original_next_race
        cfg["active_scenario"] = original_active_scenario
        cfg["active_scenario_emoji"] = original_active_emoji

@app.post("/predict-quali")
def predict_quali(params: PredictParams):
    """
    Predecir quali de la próxima carrera:

    Body opcional:
    {
      "race_name": "São Paulo Grand Prix",
      "weather_scenario": "wet"
    }
    """
    cfg = PREDICTION_CONFIG

    current_next_race = cfg["next_race"]
    current_scenario = cfg["active_scenario"]

    race_name = params.race_name or current_next_race["race_name"]
    scenario = params.weather_scenario or current_scenario

    if params.race_name and race_name not in VALID_RACE_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_race_name",
                "message": "race_name debe estar entre las opciones permitidas",
                "valid_race_names": VALID_RACE_NAMES,
            },
        )

    if params.weather_scenario and scenario not in VALID_WEATHER_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_weather_scenario",
                "message": "weather_scenario inválido",
                "valid_weather_scenarios": VALID_WEATHER_SCENARIOS,
            },
        )

    key_quali = (race_name, scenario)

    # Cache en disco
    with cache_lock:
        cached = load_cache(QUALI_CACHE_DIR, *key_quali)

    if cached is not None:
        print("Usando cache disco para", key_quali)

        return {
            **cached,
            "cached": True,
        }

    original_next_race = current_next_race.copy()
    original_active_scenario = cfg["active_scenario"]
    original_active_emoji = cfg.get("active_scenario_emoji")

    try:
        cfg["next_race"]["race_name"] = race_name
        cfg["active_scenario"] = scenario

        info = get_next_race_info(
            race_name=race_name,
            weather_scenario=scenario,
        )

        pipeline.predict_quali_next_race()
        quali_top = build_quali_top(top=20)

        response = {
            "status": "ok",
            "detail": "Predicción de quali generada",
            "cached": False,
            "next_race": info,
            "quali_predicts": quali_top,
        }

        with cache_lock:
            save_cache(QUALI_CACHE_DIR, *key_quali, {k: v for k, v in response.items() if k != "cached"})

        return response

    finally:
        cfg["next_race"] = original_next_race
        cfg["active_scenario"] = original_active_scenario
        cfg["active_scenario_emoji"] = original_active_emoji


def build_quali_top(top=10) -> List[Dict[str, Any]]:
    """
    Lee el CSV de predicciones de quali y devuelve el top n en JSON.
    Usa las columnas:
    - driver, team, pred_rank, pred_best_quali_lap_s, pred_best_quali_lap
    """
    qp = "app/models_cache/quali_predictions_latest.csv"
    
    df = pd.read_csv(qp)

    # ordenar por ranking de quali y tomar top
    df = df.sort_values("pred_rank").head(top)

    results: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        item: Dict[str, Any] = {
            "driver": r["driver"],
            "team": r["team"],
            "race_name": r.get("race_name"),
            "pred_rank": int(r["pred_rank"]),
            "pred_best_quali_lap": r["pred_best_quali_lap"],
        }
        results.append(item)

    return results



def build_race_full() -> List[Dict[str, Any]]:
    """
    Lee el CSV de predicciones de carrera y devuelve TODAS las filas en JSON.
    Usa:
    - final_position, driver, team, model_position_score,
        grid_position, predicted_position
    """
    rp = "app/models_cache/race_predictions_latest.csv"
    df = pd.read_csv(rp)

    # ordenar por posición final
    df = df.sort_values("final_position")

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        item: Dict[str, Any] = {
            "driver": r["driver"],
            "team": r["team"],
            "final_position": int(r["final_position"]),
        }
        rows.append(item)

    return rows
@app.post("/predict-all")
def predict_all(params: PredictParams):
    """
      6) Predecir quali y luego carrera usando lo predicho anteriormente.

    Permite opcionalmente indicar:
      - race_name: nombre de la carrera (debe estar en config)
      - weather_scenario: escenario meteorológico (dry, hot, wet, storm, cold)

    Devuelve:
      - top 10 de la quali
      - predicción completa de carrera (todas las posiciones)
    """
    cfg = PREDICTION_CONFIG

    current_next_race = cfg["next_race"]
    current_scenario = cfg["active_scenario"]

    race_name = params.race_name or current_next_race["race_name"]
    scenario = params.weather_scenario or current_scenario

    if params.race_name and race_name not in VALID_RACE_NAMES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_race_name",
                "message": "race_name debe estar entre las opciones definidas en config",
                "valid_race_names": VALID_RACE_NAMES,
            },
        )

    if params.weather_scenario and scenario not in VALID_WEATHER_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_weather_scenario",
                "message": "weather_scenario debe ser uno de los escenarios definidos en config",
                "valid_weather_scenarios": sorted(VALID_WEATHER_SCENARIOS),
            },
        )

    key_all = (race_name, scenario)

    # Intentar cache
    with cache_lock:
        cached = load_cache(ALL_CACHE_DIR, *key_all)

    if cached is not None:
        print("Usando cache disco para", key_all)
        return {
            **cached,
            "cached": True,
        }

    original_next_race = current_next_race.copy()
    original_active_scenario = cfg["active_scenario"]
    original_active_emoji = cfg.get("active_scenario_emoji")

    try:
        cfg["next_race"]["race_name"] = race_name
        cfg["active_scenario"] = scenario

        info = get_next_race_info(
            race_name=race_name,
            weather_scenario=scenario,
        )

        artifacts = pipeline.predict_all()

        race_full: List[Dict[str, Any]] = []
        quali_top10: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}

        try:
            race_full = build_race_full()
        except Exception as e:
            errors["race_full"] = str(e)

        try:
            quali_top10 = build_quali_top(top=10)
        except Exception as e:
            errors["quali_top10"] = str(e)

        response_core = {
            "status": "ok" if not errors else "partial_ok",
            "detail": "Predicción de quali + carrera ejecutadas",
            "next_race": info,
            "quali_top10": quali_top10,
            "race_predictions_full": race_full,
            "errors": errors or None,
        }

        response = {
            **response_core,
            "cached": False,
        }

        with cache_lock:
            save_cache(ALL_CACHE_DIR, *key_all, response_core)

        return response

    finally:
        cfg["next_race"] = original_next_race
        cfg["active_scenario"] = original_active_scenario
        cfg["active_scenario_emoji"] = original_active_emoji
