
# VisionF1 Predictions Backend

**VisionF1 Backend** is a high-performance predictive API for Formula 1 race outcomes. It leverages historical data, machine learning models (RandomForest), and real-time weather adjustments to generate accurate predictions for the 2025 season.

Built with **FastAPI**, **Pandas**, and **FastF1**, this backend powers the VisionF1 platform, providing insights into race pace, qualifying performance, and driver rankings based on evolving conditions.

---

## Key Features

*   **Advanced Prediction Pipeline**: Integrated ML models (Quali & Race) trained on 2022-2025 data.
*   **Real-Time Weather Sensitivity**: Adjusts predictions dynamically based on 5 weather scenarios (Dry, Wet, Hot, Cold, Storm).
*   **2025 Season Logic**: Full support for the 2025 grid, including rookie penalties (`ANT`, `BEA`, `BOR`, `HAD`, `LAW`) and team change factors (`HAM` to Ferrari, `SAI` to Williams, etc.).
*   **Progressive Adaptation**: Models "learn" and reduce penalties as the season progresses (race by race).
*   **Hybrid Data Source**: Combines historical stats with real-time FastF1 session data.
*   **FastAPI Performance**: Async endpoints with efficient caching (34GB+ dataset support).

---

## Tech Stack

*   **Framework**: FastAPI (Python 3.10+)
*   **Data Analysis**: Pandas, NumPy, Scikit-learn
*   **F1 Data**: FastF1 API
*   **Server**: Uvicorn (ASGI)
*   **Deployment**: Heroku / Docker ready

---

## API Endpoints

The API is served at `http://0.0.0.0:8000` by default. Documentation available at `/docs`.

### 1. Predict Race
**POST** `/predict-race`

Generates final race position predictions.

**Body:**
```json
{
  "race_name": "Abu Dhabi Grand Prix",
  "weather_scenario": "dry" 
}
```
*   `weather_scenario`: `dry` (default), `wet`, `hot`, `cold`, `storm`.

**Response:**
Returns a list of drivers with their predicted final position (`final_position`), model score (`score`), and confidence percentage (`confidence`).

---

### 2. Predict Qualifying
**POST** `/predict-quali`

Generates the predicted top 10 qualifying grid.

**Body:**
```json
{
  "race_name": "Abu Dhabi Grand Prix",
  "weather_scenario": "dry"
}
```

**Response:**
Includes `pred_rank`, `pred_best_quali_lap` (time), and `gap_to_pole`.

---

### 3. Predict All (Full Weekend)
**POST** `/predict-all`

Combined endpoint returning both Qualifying (Top 20) and Race predictions in a single call. Ideal for dashboard initialization.

**Response Structure:**
```json
{
  "next_race": { ... },
  "quali_full": [ ... 20 items ... ],
  "race_predictions_full": [ ... 20 items ... ]
}
```

---

### 4. Next Race Config
**GET** `/config-next-race`

Returns the metadata for the upcoming race (name, season, active scenario code and emoji).

---

## Weather System

VisionF1 includes a sensitivity engine (`alpha=0.5`) that modifies driver scores based on their historical performance delta in different conditions:

| Scenario | Description | Impact |
| :--- | :--- | :--- |
| **Dry** | Standard ideal conditions | Baseline performance |
| **Wet** | Rainfall and low grip | Boosts wet-weather specialists |
| **Hot** | High track temp (>50°C) | Stresses tire management |
| **Cold** | Low track temp (<15°C) | Affects tire warmup |
| **Storm** | Extreme wet conditions | High variance & penalties |

---

## 2025 Grid & Penalties

The system implements a **Progressive Adaptation Algorithm**:
*   **Rookies** (e.g., Antonelli, Bearman) start with a performance penalty that decays over 10 races.
*   **Transfers** (e.g., Hamilton to Ferrari) start with a minor adaptation penalty that decays over 5 races.
*   **Veterans** (e.g., Verstappen, Norris) retain 100% of their historical performance baseline.

---

## Installation & Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/visionf1-predictions-backend.git
    cd visionf1-predictions-backend
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the server:**
    ```bash
    fastapi run
    ```
    *Or use uvicorn directly:*
    ```bash
    python -m uvicorn main:app --reload
    ```

---

© 2025 VisionF1 Team
