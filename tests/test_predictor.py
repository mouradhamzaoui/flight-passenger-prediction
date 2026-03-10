"""
Tests — ML Predictor Core
Delta Airlines ML Platform
"""

import pytest
import numpy as np


class TestModelArtifacts:
    """Vérifie que les artifacts ML sont valides"""

    def test_model_loaded(self, model):
        assert model is not None

    def test_model_has_predict(self, model):
        assert hasattr(model, "predict"), \
            "Le modèle n'a pas de méthode predict"

    def test_feature_list_not_empty(self, feature_list):
        assert len(feature_list) > 0

    def test_training_summary_keys(self, training_summary):
        required_keys = ["best_model", "best_mae", "best_r2",
                         "best_mape", "carrier", "n_features"]
        for k in required_keys:
            assert k in training_summary, \
                f"Clé manquante dans training_summary : {k}"

    def test_carrier_is_delta(self, training_summary):
        assert training_summary["carrier"] == "DL", \
            "Le modèle n'est pas entraîné sur Delta Airlines"

    def test_mae_acceptable(self, training_summary):
        assert training_summary["best_mae"] < 5.0, \
            f"MAE trop élevée : {training_summary['best_mae']}"

    def test_r2_acceptable(self, training_summary):
        assert training_summary["best_r2"] > 0.90, \
            f"R² insuffisant : {training_summary['best_r2']}"

    def test_encoders_have_route(self, encoders):
        assert "route"  in encoders
        assert "origin" in encoders
        assert "dest"   in encoders


class TestPredictorSingleton:
    """Vérifie le predictor singleton"""

    def test_predictor_loads(self):
        from backend.app.core.predictor import predictor
        predictor.load()
        assert predictor._loaded is True

    def test_predict_flight_returns_dict(self):
        from backend.app.core.predictor import predictor
        predictor.load()
        result = predictor.predict_flight(
            origin="ATL", dest="LAX", year=2024, month=7,
            day_of_week=5, seats=180, avg_ticket_price=320,
            weather="CLEAR", is_holiday=True
        )
        assert isinstance(result, dict)

    def test_predict_flight_load_factor_range(self):
        from backend.app.core.predictor import predictor
        predictor.load()
        result = predictor.predict_flight(
            origin="ATL", dest="LAX", year=2024, month=7,
            day_of_week=5, seats=180, avg_ticket_price=320,
            weather="CLEAR", is_holiday=True
        )
        lf = result["predicted_load_factor"]
        assert 20 <= lf <= 100, f"Load Factor hors range : {lf}"

    def test_predict_flight_carrier_is_dl(self):
        from backend.app.core.predictor import predictor
        predictor.load()
        result = predictor.predict_flight(
            origin="ATL", dest="LAX", year=2024, month=7,
            day_of_week=5, seats=180, avg_ticket_price=320,
            weather="CLEAR", is_holiday=False
        )
        assert result["carrier"] == "DL"

    def test_predict_flight_passengers_consistent(self):
        from backend.app.core.predictor import predictor
        predictor.load()
        seats = 180
        result = predictor.predict_flight(
            origin="ATL", dest="LAX", year=2024, month=7,
            day_of_week=5, seats=seats, avg_ticket_price=320,
            weather="CLEAR", is_holiday=False
        )
        pax = result["predicted_passengers"]
        lf  = result["predicted_load_factor"]
        expected_pax = int(seats * lf / 100)
        assert abs(pax - expected_pax) <= 2, \
            f"Incohérence pax/lf : {pax} vs {expected_pax}"

    def test_predict_route_yearly_has_12_months(self):
        from backend.app.core.predictor import predictor
        predictor.load()
        result = predictor.predict_route_yearly("ATL", "LAX", 2024)
        assert len(result["monthly_forecasts"]) == 12

    def test_summer_higher_than_winter(self):
        from backend.app.core.predictor import predictor
        predictor.load()
        summer = predictor.predict_flight(
            "ATL","LAX",2024,7,5,180,300,"CLEAR",False
        )["predicted_load_factor"]
        winter = predictor.predict_flight(
            "ATL","LAX",2024,1,5,180,300,"CLEAR",False
        )["predicted_load_factor"]
        assert summer >= winter, \
            f"Été ({summer:.1f}%) < Hiver ({winter:.1f}%) — logique incorrecte"