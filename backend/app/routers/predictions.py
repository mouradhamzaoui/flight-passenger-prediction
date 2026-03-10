"""
API Routers — Delta Airlines Predictions
Endpoints: /predict/flight | /predict/route | /predict/airport
"""

from fastapi import APIRouter, HTTPException
from backend.app.schemas.prediction import (
    FlightPredictionInput, FlightPredictionOutput,
    RoutePredictionInput, RouteAnalysisOutput,
    AirportPredictionInput, AirportSummaryOutput,
)
from backend.app.core.predictor import predictor
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions — Delta Airlines"])


@router.post("/flight", response_model=FlightPredictionOutput,
             summary="Predict Load Factor for a single Delta flight")
async def predict_flight(payload: FlightPredictionInput):
    try:
        result = predictor.predict_flight(
            origin=payload.origin.value,
            dest=payload.dest.value,
            year=payload.year,
            month=payload.month,
            day_of_week=payload.day_of_week,
            seats=payload.seats,
            avg_ticket_price=payload.avg_ticket_price,
            weather=payload.weather_condition.value,
            is_holiday=payload.is_holiday_period,
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/route", response_model=RouteAnalysisOutput,
             summary="Predict monthly Load Factor for a Delta route (full year)")
async def predict_route(payload: RoutePredictionInput):
    try:
        return predictor.predict_route_yearly(
            payload.origin.value,
            payload.dest.value,
            payload.year,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/airport", response_model=AirportSummaryOutput,
             summary="Predict Load Factor for all routes at a Delta hub airport")
async def predict_airport(payload: AirportPredictionInput):
    try:
        return predictor.predict_airport(
            payload.airport.value,
            payload.month,
            payload.year,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
