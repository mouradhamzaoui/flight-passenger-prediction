"""
Pytest Configuration — Delta Airlines ML Platform
Fixtures partagées entre tous les tests
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from fastapi.testclient import TestClient

# ─── FIXTURES CHEMINS ─────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def models_dir():
    return Path("data/models")

@pytest.fixture(scope="session")
def features_dir():
    return Path("data/features")

# ─── FIXTURES ARTIFACTS ───────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def model(models_dir):
    with open(models_dir / "best_model.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def feature_list(models_dir):
    with open(models_dir / "feature_list.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def training_summary(models_dir):
    with open(models_dir / "training_summary.json") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def encoders(features_dir):
    with open(features_dir / "label_encoders.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def df_full(features_dir):
    return pd.read_csv(features_dir / "delta_features_full.csv")

@pytest.fixture(scope="session")
def df_ml(features_dir):
    return pd.read_csv(features_dir / "delta_features_ml.csv")

# ─── FIXTURE FASTAPI CLIENT ───────────────────────────────────────────────────
@pytest.fixture(scope="session")
def api_client():
    from backend.app.main import app
    from backend.app.core.predictor import predictor
    predictor.load()
    return TestClient(app)

# ─── FIXTURE PAYLOAD TYPE ─────────────────────────────────────────────────────
@pytest.fixture
def valid_flight_payload():
    return {
        "origin":            "ATL",
        "dest":              "LAX",
        "year":              2024,
        "month":             7,
        "day_of_week":       5,
        "seats":             180,
        "avg_ticket_price":  320.0,
        "weather_condition": "CLEAR",
        "is_holiday_period": True
    }

@pytest.fixture
def valid_route_payload():
    return {"origin": "ATL", "dest": "LAX", "year": 2024}

@pytest.fixture
def valid_airport_payload():
    return {"airport": "ATL", "month": 7, "year": 2024}