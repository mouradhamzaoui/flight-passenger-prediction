"""
ML Predictor Core — Delta Airlines
Singleton pattern — chargement unique au démarrage
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = Path("data/models")
FEATURES_DIR = Path("data/features")

HUB_TIER = {
    "ATL": 1, "DTW": 1, "MSP": 1, "SLC": 2, "SEA": 2,
    "BOS": 2, "LGA": 2, "LAX": 2, "JFK": 2, "MCO": 3, "MIA": 3
}
DISTANCES = {
    ("ATL", "LGA"): 762, ("ATL", "BOS"): 1099, ("ATL", "LAX"): 1946,
    ("ATL", "MCO"): 403, ("ATL", "MIA"): 662, ("ATL", "DTW"): 594,
    ("ATL", "MSP"): 907, ("ATL", "SLC"): 1589, ("ATL", "SEA"): 2182,
    ("ATL", "JFK"): 760, ("DTW", "MSP"): 528, ("DTW", "BOS"): 632,
    ("DTW", "LGA"): 502, ("MSP", "SLC"): 987, ("MSP", "SEA"): 1399,
    ("SLC", "SEA"): 689, ("SEA", "LAX"): 954, ("SEA", "JFK"): 2422,
    ("LGA", "MCO"): 1074, ("BOS", "MCO"): 1123,
}
# Distances bidirectionnelles
for (o, d), v in list(DISTANCES.items()):
    DISTANCES[(d, o)] = v

SEASON_IDX = {1: 0.88, 2: 0.85, 3: 0.95, 4: 0.97, 5: 1.00, 6: 1.08,
              7: 1.12, 8: 1.10, 9: 0.92, 10: 0.94, 11: 1.02, 12: 1.05}
WEATHER_MAP = {"CLEAR": 0, "CLOUDY": 1, "RAIN": 2, "SNOW": 3}


class DeltaPredictor:
    """Singleton ML predictor — Delta Airlines Load Factor"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        logger.info("Loading Delta Airlines ML artifacts...")
        with open(MODELS_DIR / "best_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(MODELS_DIR / "feature_list.pkl", "rb") as f:
            self.features = pickle.load(f)
        with open(MODELS_DIR / "training_summary.json") as f:
            self.summary = json.load(f)
        with open(FEATURES_DIR / "label_encoders.pkl", "rb") as f:
            self.encoders = pickle.load(f)

        self.df_full = pd.read_csv(FEATURES_DIR / "delta_features_full.csv")
        self.df_ml = pd.read_csv(FEATURES_DIR / "delta_features_ml.csv")
        self.medians = self.df_ml[self.features].median()

        self._loaded = True
        logger.info(f"Model loaded: {self.summary['best_model']} "
                    f"| MAE={self.summary['best_mae']:.3f}%")

    def _build_input(self, origin: str, dest: str, year: int, month: int,
                     day_of_week: int, seats: int, avg_ticket_price: float,
                     weather: str, is_holiday: bool) -> pd.DataFrame:

        dist = DISTANCES.get((origin, dest),
                             DISTANCES.get((dest, origin), 1000))

        mask = (
            self.df_full["origin"] == origin) & (
            self.df_full["dest"] == dest)
        r_avg_lf = self.df_full[mask]["load_factor"].mean() \
            if mask.sum() > 0 else self.df_full["load_factor"].mean()
        r_std_lf = self.df_full[mask]["load_factor"].std() \
            if mask.sum() > 0 else 8.0
        r_std_lf = r_std_lf if not np.isnan(r_std_lf) else 8.0

        route_key = f"{origin}_{dest}"
        if "route" not in self.df_full.columns:
            self.df_full["route"] = \
                self.df_full["origin"] + "_" + self.df_full["dest"]
        route_pop = self.df_full.groupby("route")["passengers"].sum()
        r_pop_rank = int(route_pop.rank(ascending=False).get(route_key, 10))
        net_avg_lf = self.df_full[
            self.df_full["month"] == month]["load_factor"].mean()
        net_avg_price = self.df_full[
            self.df_full["month"] == month]["avg_ticket_price"].mean()

        try:
            r_enc = self.encoders["route"].transform([route_key])[0]
            o_enc = self.encoders["origin"].transform([origin])[0]
            d_enc = self.encoders["dest"].transform([dest])[0]
        except Exception:
            r_enc = o_enc = d_enc = 0

        m = self.medians
        inp = {
            "year": year, "month": month, "day_of_week": day_of_week,
            "quarter": (month - 1) // 3 + 1,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
            "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
            "is_weekend": int(day_of_week >= 6),
            "is_monday": int(day_of_week == 1),
            "is_friday": int(day_of_week == 5),
            "is_summer": int(month in [6, 7, 8]),
            "is_winter": int(month in [12, 1, 2]),
            "is_spring": int(month in [3, 4, 5]),
            "is_fall": int(month in [9, 10, 11]),
            "is_peak_travel": int(month in [6, 7, 8, 11, 12]),
            "is_holiday_period": int(is_holiday),
            "is_thanksgiving": int(month == 11 and is_holiday),
            "is_xmas_newyear": int(month == 12 and is_holiday),
            "is_july4": int(month == 7 and is_holiday),
            "distance": dist, "seats": seats,
            "avg_ticket_price": avg_ticket_price,
            "price_per_mile": avg_ticket_price / max(dist, 1),
            "revenue_per_seat": seats * 0.85 * avg_ticket_price / max(seats, 1),
            "yield_metric": avg_ticket_price / max(dist, 1) * 100,
            "is_long_haul": int(dist >= 1500),
            "is_medium_haul": int(500 <= dist < 1500),
            "is_short_haul": int(dist < 500),
            "weather_encoded": WEATHER_MAP.get(weather, 0),
            "route_avg_lf": r_avg_lf, "route_std_lf": r_std_lf,
            "route_max_lf": self.df_full["load_factor"].max(),
            "route_avg_price": avg_ticket_price,
            "route_avg_distance": dist,
            "route_lf_cv": r_std_lf / max(r_avg_lf, 1),
            "route_popularity_rank": r_pop_rank,
            "route_encoded": r_enc,
            "origin_hub_tier": HUB_TIER.get(origin, 3),
            "dest_hub_tier": HUB_TIER.get(dest, 3),
            "hub_to_hub": int(HUB_TIER.get(origin, 3) == 1 and
                              HUB_TIER.get(dest, 3) == 1),
            "hub_to_spoke": int(HUB_TIER.get(origin, 3) == 1 and
                                HUB_TIER.get(dest, 3) >= 2),
            "is_atl_flight": int(origin == "ATL" or dest == "ATL"),
            "origin_avg_lf": self.df_full[
                self.df_full["origin"] == origin]["load_factor"].mean()
            if len(self.df_full[self.df_full["origin"] == origin]) > 0
            else self.df_full["load_factor"].mean(),
            "origin_n_routes": int(self.df_full[
                self.df_full["origin"] == origin]["dest"].nunique())
            if len(self.df_full[self.df_full["origin"] == origin]) > 0 else 5,
            "origin_encoded": o_enc, "dest_encoded": d_enc,
            "network_avg_lf": net_avg_lf, "network_avg_price": net_avg_price,
            "price_vs_network_avg": avg_ticket_price - net_avg_price,
            "is_post_covid": int(year >= 2022),
            "is_covid_period": 0, "recovery_phase": 0,
            "covid_impact_factor": 1.0,
            "seasonality_index": SEASON_IDX.get(month, 1.0),
            "lf_lag_1m": m.get("lf_lag_1m", r_avg_lf),
            "lf_lag_2m": m.get("lf_lag_2m", r_avg_lf),
            "lf_lag_3m": m.get("lf_lag_3m", r_avg_lf),
            "lf_lag_6m": m.get("lf_lag_6m", r_avg_lf),
            "lf_lag_12m": m.get("lf_lag_12m", r_avg_lf),
            "lf_rolling_mean_3m": m.get("lf_rolling_mean_3m", r_avg_lf),
            "lf_rolling_mean_6m": m.get("lf_rolling_mean_6m", r_avg_lf),
            "lf_rolling_mean_12m": m.get("lf_rolling_mean_12m", r_avg_lf),
            "lf_rolling_std_3m": m.get("lf_rolling_std_3m", 8.0),
            "lf_rolling_std_6m": m.get("lf_rolling_std_6m", 8.0),
            "lf_mom_change": 0.0, "lf_yoy_change": 0.0,
        }
        return pd.DataFrame([inp])[self.features].fillna(0)

    def predict_flight(self, origin, dest, year, month, day_of_week,
                       seats, avg_ticket_price, weather, is_holiday) -> dict:
        X = self._build_input(origin, dest, year, month, day_of_week,
                              seats, avg_ticket_price, weather, is_holiday)
        pred = float(np.clip(self.model.predict(X)[0], 20, 100))
        mae = self.summary["best_mae"]
        pax = int(seats * pred / 100)
        rev = round(pax * avg_ticket_price, 2)

        if pred >= 85:
            rating = "EXCELLENT"
        elif pred >= 70:
            rating = "GOOD"
        elif pred >= 55:
            rating = "AVERAGE"
        else:
            rating = "LOW"

        return {
            "carrier": "DL",
            "origin": origin,
            "dest": dest,
            "route": f"{origin}-{dest}",
            "month": month,
            "year": year,
            "predicted_load_factor": round(pred, 2),
            "predicted_passengers": pax,
            "estimated_revenue": rev,
            "confidence_band_low": round(max(pred - mae * 2, 0), 2),
            "confidence_band_high": round(min(pred + mae * 2, 100), 2),
            "performance_rating": rating,
            "model_used": self.summary["best_model"],
            "mae_model": mae,
        }

    def predict_route_yearly(self, origin, dest, year) -> dict:
        monthly = []
        for month in range(1, 13):
            r = self.predict_flight(
                origin, dest, year, month,
                day_of_week=5, seats=180,
                avg_ticket_price=250,
                weather="CLEAR", is_holiday=False
            )
            monthly.append({
                "month": month,
                "predicted_lf": r["predicted_load_factor"],
                "predicted_pax": r["predicted_passengers"],
                "performance_rating": r["performance_rating"],
            })
        avg_lf = round(sum(m["predicted_lf"] for m in monthly) / 12, 2)
        best_month = max(monthly, key=lambda x: x["predicted_lf"])["month"]
        worst_month = min(monthly, key=lambda x: x["predicted_lf"])["month"]
        return {
            "carrier": "DL",
            "origin": origin,
            "dest": dest,
            "route": f"{origin}-{dest}",
            "year": year,
            "monthly_forecasts": monthly,
            "avg_predicted_lf": avg_lf,
            "best_month": best_month,
            "worst_month": worst_month,
            "model_used": self.summary["best_model"],
        }

    def predict_airport(self, airport, month, year) -> dict:
        routes_out = [
            (airport, d) for (o, d) in DISTANCES
            if o == airport and d != airport
        ][:8]
        summaries = []
        total_pax = 0
        lf_vals = []
        for o, d in routes_out:
            r = self.predict_flight(
                o, d, year, month,
                day_of_week=5, seats=180,
                avg_ticket_price=250,
                weather="CLEAR", is_holiday=False
            )
            summaries.append({
                "route": f"{o}-{d}",
                "predicted_lf": r["predicted_load_factor"],
                "predicted_pax": r["predicted_passengers"],
                "rating": r["performance_rating"],
            })
            total_pax += r["predicted_passengers"]
            lf_vals.append(r["predicted_load_factor"])

        return {
            "carrier": "DL",
            "airport": airport,
            "month": month,
            "year": year,
            "routes_summary": summaries,
            "airport_avg_lf": round(sum(lf_vals) / max(len(lf_vals), 1), 2),
            "total_pred_pax": total_pax,
            "model_used": self.summary["best_model"],
        }


predictor = DeltaPredictor()
