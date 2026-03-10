"""
Pydantic Schemas — Delta Airlines ML API
Input validation + Response models
"""

from pydantic import BaseModel, Field, field_validator
from typing   import Optional, List
from enum     import Enum

class WeatherCondition(str, Enum):
    CLEAR  = "CLEAR"
    CLOUDY = "CLOUDY"
    RAIN   = "RAIN"
    SNOW   = "SNOW"

class DeltaOrigin(str, Enum):
    ATL = "ATL"; DTW = "DTW"; MSP = "MSP"; SLC = "SLC"
    SEA = "SEA"; BOS = "BOS"; LGA = "LGA"; LAX = "LAX"
    JFK = "JFK"; MCO = "MCO"; MIA = "MIA"

class DeltaDest(str, Enum):
    ATL = "ATL"; DTW = "DTW"; MSP = "MSP"; SLC = "SLC"
    SEA = "SEA"; BOS = "BOS"; LGA = "LGA"; LAX = "LAX"
    JFK = "JFK"; MCO = "MCO"; MIA = "MIA"

# ─── INPUT ────────────────────────────────────────────────────────────────────
class FlightPredictionInput(BaseModel):
    origin:            DeltaOrigin
    dest:              DeltaDest
    year:              int   = Field(2024, ge=2019, le=2030)
    month:             int   = Field(...,  ge=1,    le=12)
    day_of_week:       int   = Field(5,    ge=1,    le=7)
    seats:             int   = Field(180,  ge=50,   le=400)
    avg_ticket_price:  float = Field(220,  ge=20,   le=2000)
    weather_condition: WeatherCondition = WeatherCondition.CLEAR
    is_holiday_period: bool  = False

    @field_validator("dest")
    @classmethod
    def origin_dest_different(cls, dest, info):
        if "origin" in info.data and dest == info.data["origin"]:
            raise ValueError("Origin and destination must be different")
        return dest

    model_config = {
        "json_schema_extra": {
            "example": {
                "origin": "ATL", "dest": "LAX",
                "year": 2024, "month": 7, "day_of_week": 5,
                "seats": 180, "avg_ticket_price": 320,
                "weather_condition": "CLEAR", "is_holiday_period": True
            }
        }
    }

class RoutePredictionInput(BaseModel):
    origin: DeltaOrigin
    dest:   DeltaDest
    year:   int = Field(2024, ge=2019, le=2030)

class AirportPredictionInput(BaseModel):
    airport: DeltaOrigin
    month:   int = Field(..., ge=1, le=12)
    year:    int = Field(2024, ge=2019, le=2030)

# ─── OUTPUT ───────────────────────────────────────────────────────────────────
class FlightPredictionOutput(BaseModel):
    carrier:              str
    origin:               str
    dest:                 str
    route:                str
    month:                int
    year:                 int
    predicted_load_factor: float
    predicted_passengers:  int
    estimated_revenue:     float
    confidence_band_low:   float
    confidence_band_high:  float
    performance_rating:    str
    model_used:            str
    mae_model:             float

class RouteAnalysisOutput(BaseModel):
    carrier:          str
    origin:           str
    dest:             str
    route:            str
    year:             int
    monthly_forecasts: List[dict]
    avg_predicted_lf:  float
    best_month:        int
    worst_month:       int
    model_used:        str

class AirportSummaryOutput(BaseModel):
    carrier:       str
    airport:       str
    month:         int
    year:          int
    routes_summary: List[dict]
    airport_avg_lf: float
    total_pred_pax: int
    model_used:     str

class HealthResponse(BaseModel):
    status:     str
    carrier:    str
    model:      str
    mae:        float
    r2:         float
    n_features: int
    version:    str