"""
Tests — FastAPI Endpoints
Delta Airlines ML Platform
"""

import pytest


class TestHealthEndpoints:
    """Tests endpoints santé"""

    def test_root_returns_200(self, api_client):
        r = api_client.get("/")
        assert r.status_code == 200

    def test_root_carrier_is_dl(self, api_client):
        r = api_client.get("/")
        assert r.json()["carrier"] == "DL"

    def test_health_returns_200(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_health_status_healthy(self, api_client):
        r = api_client.get("/health")
        assert r.json()["status"] == "healthy"

    def test_health_mae_acceptable(self, api_client):
        r = api_client.get("/health")
        assert r.json()["mae"] < 5.0

    def test_health_r2_acceptable(self, api_client):
        r = api_client.get("/health")
        assert r.json()["r2"] > 0.90


class TestFlightPrediction:
    """Tests endpoint /predict/flight"""

    def test_predict_flight_status_200(self, api_client, valid_flight_payload):
        r = api_client.post("/predict/flight", json=valid_flight_payload)
        assert r.status_code == 200, r.text

    def test_predict_flight_has_load_factor(self, api_client, valid_flight_payload):
        r = api_client.post("/predict/flight", json=valid_flight_payload)
        assert "predicted_load_factor" in r.json()

    def test_predict_flight_lf_in_range(self, api_client, valid_flight_payload):
        r = api_client.post("/predict/flight", json=valid_flight_payload)
        lf = r.json()["predicted_load_factor"]
        assert 20 <= lf <= 100, f"LF hors range : {lf}"

    def test_predict_flight_carrier_dl(self, api_client, valid_flight_payload):
        r = api_client.post("/predict/flight", json=valid_flight_payload)
        assert r.json()["carrier"] == "DL"

    def test_predict_flight_correct_route(self, api_client, valid_flight_payload):
        r = api_client.post("/predict/flight", json=valid_flight_payload)
        assert r.json()["origin"] == "ATL"
        assert r.json()["dest"]   == "LAX"

    def test_predict_flight_has_confidence_band(self, api_client, valid_flight_payload):
        r    = api_client.post("/predict/flight", json=valid_flight_payload)
        data = r.json()
        assert data["confidence_band_low"]  <= data["predicted_load_factor"]
        assert data["confidence_band_high"] >= data["predicted_load_factor"]

    def test_predict_flight_same_origin_dest_rejected(self, api_client):
        payload = {"origin":"ATL","dest":"ATL","year":2024,
                   "month":7,"day_of_week":5,"seats":180,
                   "avg_ticket_price":300.0,
                   "weather_condition":"CLEAR","is_holiday_period":False}
        r = api_client.post("/predict/flight", json=payload)
        assert r.status_code == 422

    def test_predict_flight_invalid_month_rejected(self, api_client):
        payload = {"origin":"ATL","dest":"LAX","year":2024,
                   "month":13,"day_of_week":5,"seats":180,
                   "avg_ticket_price":300.0,
                   "weather_condition":"CLEAR","is_holiday_period":False}
        r = api_client.post("/predict/flight", json=payload)
        assert r.status_code == 422


class TestRoutePrediction:
    """Tests endpoint /predict/route"""

    def test_predict_route_status_200(self, api_client, valid_route_payload):
        r = api_client.post("/predict/route", json=valid_route_payload)
        assert r.status_code == 200, r.text

    def test_predict_route_has_12_months(self, api_client, valid_route_payload):
        r = api_client.post("/predict/route", json=valid_route_payload)
        assert len(r.json()["monthly_forecasts"]) == 12

    def test_predict_route_carrier_dl(self, api_client, valid_route_payload):
        r = api_client.post("/predict/route", json=valid_route_payload)
        assert r.json()["carrier"] == "DL"

    def test_predict_route_has_best_worst_month(self, api_client, valid_route_payload):
        r    = api_client.post("/predict/route", json=valid_route_payload)
        data = r.json()
        assert 1 <= data["best_month"]  <= 12
        assert 1 <= data["worst_month"] <= 12


class TestAirportPrediction:
    """Tests endpoint /predict/airport"""

    def test_predict_airport_status_200(self, api_client, valid_airport_payload):
        r = api_client.post("/predict/airport", json=valid_airport_payload)
        assert r.status_code == 200, r.text

    def test_predict_airport_carrier_dl(self, api_client, valid_airport_payload):
        r = api_client.post("/predict/airport", json=valid_airport_payload)
        assert r.json()["carrier"] == "DL"

    def test_predict_airport_has_routes(self, api_client, valid_airport_payload):
        r = api_client.post("/predict/airport", json=valid_airport_payload)
        assert len(r.json()["routes_summary"]) > 0

    def test_predict_airport_lf_in_range(self, api_client, valid_airport_payload):
        r  = api_client.post("/predict/airport", json=valid_airport_payload)
        lf = r.json()["airport_avg_lf"]
        assert 20 <= lf <= 100