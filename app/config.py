# Configuración del rango de carreras para el entrenamiento - MÚLTIPLES AÑOS
RACE_RANGE = {
    "years": [2022, 2023, 2024, 2025],  # Años a descargar
    "max_races_per_year": 24,  # Máximo de carreras por año (F1 tiene ~24 carreras)
    "include_current_year": True,  # Incluir año actual aunque esté incompleto
    "auto_detect_available": True,  # Detectar automáticamente carreras disponibles
    "stop_on_future_races": True   # Parar cuando encuentre carreras futuras
}








"""
=== Nombres de carreras detectados ===
- Abu Dhabi Grand Prix
- Australian Grand Prix
- Austrian Grand Prix
- Azerbaijan Grand Prix
- Bahrain Grand Prix
- Belgian Grand Prix
- British Grand Prix
- Canadian Grand Prix
- Chinese Grand Prix
- Dutch Grand Prix
- Emilia Romagna Grand Prix
- French Grand Prix
- Hungarian Grand Prix
- Italian Grand Prix
- Japanese Grand Prix
- Las Vegas Grand Prix
- Mexico City Grand Prix
- Miami Grand Prix
- Monaco Grand Prix
- Qatar Grand Prix
- Saudi Arabian Grand Prix
- Singapore Grand Prix
- Spanish Grand Prix
- São Paulo Grand Prix
- United States Grand Prix


"""




# Configuración para predicción de próxima carrera
PREDICTION_CONFIG = {
    "next_race": {
        "year": 2025,
        "race_name": "Abu Dhabi Grand Prix", 
        "circuit_name": "Yas Marina Circuit",
        "race_number": 24  # Número de carrera en la temporada 2025
    },
    "use_historical_data": True,
    
    # 🌤️ CONFIGURACIÓN METEOROLÓGICA PARA PREDICCIONES
    "weather_scenarios": {
        
        # Escenario seco - condiciones ideales
        "dry": {
            "session_air_temp": 26.0,      # Temperatura ideal
            "session_track_temp": 35.0,    # Temperatura de pista normal
            "session_humidity": 45.0,      # Humedad baja
            "session_rainfall": False,     # Sin lluvia
            "description": "Condiciones secas e ideales"
        },
        
        # Escenario caluroso - estrés térmico
        "hot": {
            "session_air_temp": 35.0,      # Muy caluroso
            "session_track_temp": 50.0,    # Pista muy caliente
            "session_humidity": 70.0,      # Humedad alta = más estrés
            "session_rainfall": False,     # Sin lluvia
            "description": "Condiciones muy calurosas (estrés térmico)"
        },
        
        # Escenario húmedo - lluvia ligera
        "wet": {
            "session_air_temp": 18.0,      # Más fresco por lluvia
            "session_track_temp": 22.0,    # Pista fría
            "session_humidity": 85.0,      # Muy húmedo
            "session_rainfall": True,      # Lluvia confirmada
            "description": "Condiciones húmedas con lluvia"
        },
        
        # Escenario extremo - tormenta
        "storm": {
            "session_air_temp": 15.0,      # Frío
            "session_track_temp": 18.0,    # Pista muy fría
            "session_humidity": 95.0,      # Humedad extrema
            "session_rainfall": True,      # Lluvia intensa
            "description": "Condiciones extremas - tormenta"
        },
        
        # Escenario frío - condiciones invernales
        "cold": {
            "session_air_temp": 12.0,      # Muy frío
            "session_track_temp": 15.0,    # Pista fría
            "session_humidity": 60.0,      # Humedad media
            "session_rainfall": False,     # Seco pero frío
            "description": "Condiciones muy frías"
        }
    },
    
    # 🎯 CONFIGURACIÓN DE PREDICCIÓN ACTIVA
    "active_scenario": "dry",  # Cambiar por: "dry", "hot", "wet", "storm", "cold"
    "active_scenario_emoji": "☀️"
    # "dry": "☀️",
    # "hot": "🔥",
    # "wet": "🌧️",
    # "storm": "⛈️",
    # "cold": "❄️"


}

# PESOS POR AÑOS - Importancia temporal de los datos
DATA_IMPORTANCE = {
    "2025_weight": 0.50,  # 50% - Datos más recientes (máxima importancia)
    "2024_weight": 0.30,  # 30% - Año anterior (alta importancia)
    "2023_weight": 0.15,  # 15% - Hace 2 años (media importancia)
    "2022_weight": 0.05,  # 5% - Hace 3 años (baja importancia)
}

# Solo pilotos activos 2025
DRIVERS_2025 = {
    # Red Bull
    "VER": {"team": "Red Bull Racing"},
    "TSU": {"team": "Red Bull Racing", "team_change": True},

    # Ferrari
    "LEC": {"team": "Ferrari"},
    "HAM": {"team": "Ferrari", "team_change": True},
    
    # McLaren
    "NOR": {"team": "McLaren"},
    "PIA": {"team": "McLaren"},
    
    
    # Mercedes
    "RUS": {"team": "Mercedes"},
    "ANT": {"team": "Mercedes", "rookie": True},
    
    # Williams
    "ALB": {"team": "Williams"},
    "SAI": {"team": "Williams", "team_change": True},
    
    # Racing Bulls
    "HAD": {"team": "Racing Bulls", "rookie": True},
    "LAW": {"team": "Racing Bulls", "rookie": True},
    
    # Aston Martin
    "ALO": {"team": "Aston Martin"},
    "STR": {"team": "Aston Martin"},
    
    # Haas
    "OCO": {"team": "Haas", "team_change": True},
    "BEA": {"team": "Haas", "rookie": True},
    
    # Alpine
    "GAS": {"team": "Alpine"},
    "COL": {"team": "Alpine", "team_change": True},
    
    # Sauber
    "HUL": {"team": "Sauber"},
    "BOR": {"team": "Sauber", "rookie": True}
}

# 🔥 PENALIZACIONES SIMPLES
PENALTIES = {
    "rookie": 2.5,           # Penalización para rookies
    "team_change": 1.5,      # Penalización por cambio de equipo
    "adaptation_races": 10,   # Carreras para adaptarse completamente
    "use_progressive": True  # Usar sistema de adaptación progresiva
}

# Listas simples
ROOKIES_2025 = ["ANT", "BEA", "BOR", "HAD", "LAW"]
RETIRED_DRIVERS = ["PER", "MAG", "DOO", "RIC", "BOT", "ZHO", "SAR"]



VALID_TEAMS = ['Alpine', 'Aston Martin', 'Ferrari', 'Haas F1 Team', 'Kick Sauber', 
                      'McLaren', 'Mercedes', 'Racing Bulls', 'Red Bull Racing', 'Williams']
        

# 🔥 CONFIGURACIÓN SIMPLE DE ADAPTACIÓN
ADAPTATION_SYSTEM = {
    "change_types": {
        "rookie": {
            "base_penalty": PENALTIES["rookie"],
            "adaptation_races": PENALTIES["adaptation_races"],
            "description": "Piloto completamente nuevo"
        },
        "team_change": {
            "base_penalty": PENALTIES["team_change"],
            "adaptation_races": 5,
            "description": "Cambio de equipo"
        }
    }
}

# 🔥 FACTORES DE AJUSTE CONSOLIDADOS
ADJUSTMENT_FACTORS = {
    "use_progressive_adaptation": PENALTIES["use_progressive"]
}

# ⚙️ CONFIGURACIÓN DE PREDICCIÓN DE CARRERA
RACE_PREDICTION = {
    # Peso de la grilla en la mezcla carrera = (1-β)*modelo + β*grilla
    "grid_mix_beta": 0.35
}


SCENARIO_EMOJIS = {
    "dry": "☀️",
    "hot": "🔥",
    "wet": "🌧️",
    "storm": "⛈️",
    "cold": "❄️",
}
