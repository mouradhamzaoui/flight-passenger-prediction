"""
FastAPI Main — Delta Airlines ML Platform
Production-Ready API | Standard Airbus/Amadeus 2026
"""

from fastapi                    import FastAPI
from fastapi.middleware.cors    import CORSMiddleware
from contextlib                 import asynccontextmanager
from backend.app.routers        import predictions
from backend.app.core.predictor import predictor
from backend.app.schemas.prediction import HealthResponse
import logging

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Delta Airlines ML API...")
    predictor.load()
    logger.info("ML artifacts loaded — API ready")
    yield
    logger.info("Shutting down Delta Airlines ML API")

app = FastAPI(
    title       = "Delta Airlines — Load Factor Prediction API",
    description = """
## ✈️ Delta Air Lines ML Platform

Production-ready REST API for **flight passenger demand forecasting**.

### Endpoints
| Endpoint | Description |
|----------|-------------|
| `POST /predict/flight`   | Load Factor for a single flight |
| `POST /predict/route`    | Monthly forecast for a full year |
| `POST /predict/airport`  | All routes summary for a hub |

### Carrier
> **DL — Delta Air Lines only**

### Model
> LightGBM / XGBoost — R² > 0.999 — MAE < 0.5%

### Standard
> Airbus / Amadeus MLOps 2026
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(predictions.router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Delta Airlines ML Load Factor Prediction API",
        "carrier": "DL",
        "docs":    "/docs",
        "version": "1.0.0",
        "status":  "operational",
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    s = predictor.summary
    return {
        "status":     "healthy",
        "carrier":    "DL",
        "model":      s["best_model"],
        "mae":        s["best_mae"],
        "r2":         s["best_r2"],
        "n_features": s["n_features"],
        "version":    "1.0.0",
    }