"""
Tests — Feature Engineering
Delta Airlines ML Platform
"""

import pytest
import pandas as pd
import numpy as np


class TestDatasetIntegrity:
    """Vérifie l'intégrité du dataset Delta Airlines"""

    def test_dataset_not_empty(self, df_full):
        assert len(df_full) > 0, "Dataset vide"

    def test_carrier_is_delta_only(self, df_full):
        carriers = df_full["unique_carrier"].unique()
        assert list(carriers) == ["DL"], \
            f"Dataset contient d'autres compagnies : {carriers}"

    def test_load_factor_range(self, df_full):
        assert df_full["load_factor"].between(0, 100).all(), \
            "Load Factor hors range [0, 100]"

    def test_no_negative_passengers(self, df_full):
        assert (df_full["passengers"] >= 0).all(), \
            "Passagers négatifs détectés"

    def test_no_negative_seats(self, df_full):
        assert (df_full["seats"] > 0).all(), \
            "Nombre de sièges invalide"

    def test_years_range(self, df_full):
        assert df_full["year"].between(2019, 2023).all(), \
            "Années hors plage 2019-2023"

    def test_month_range(self, df_full):
        assert df_full["month"].between(1, 12).all(), \
            "Mois hors plage 1-12"

    def test_required_columns_exist(self, df_full):
        required = ["year", "month", "unique_carrier", "origin",
                    "dest", "load_factor", "passengers", "seats",
                    "avg_ticket_price", "distance"]
        missing = [c for c in required if c not in df_full.columns]
        assert not missing, f"Colonnes manquantes : {missing}"

    def test_delta_hubs_present(self, df_full):
        hubs = {"ATL", "DTW", "MSP", "SLC", "SEA"}
        origins = set(df_full["origin"].unique())
        assert hubs.issubset(origins), \
            f"Hubs Delta manquants : {hubs - origins}"


class TestMLDataset:
    """Vérifie le dataset ML final"""

    def test_ml_dataset_not_empty(self, df_ml):
        assert len(df_ml) > 10_000, \
            f"Dataset ML trop petit : {len(df_ml)} lignes"

    def test_target_exists(self, df_ml):
        assert "load_factor" in df_ml.columns, \
            "Colonne target 'load_factor' manquante"

    def test_no_all_nan_columns(self, df_ml):
        all_nan = df_ml.columns[df_ml.isna().all()].tolist()
        assert not all_nan, f"Colonnes 100% NaN : {all_nan}"

    def test_feature_count(self, feature_list):
        assert len(feature_list) >= 60, \
            f"Nombre de features insuffisant : {len(feature_list)}"

    def test_cyclic_features_exist(self, feature_list):
        assert "month_sin" in feature_list
        assert "month_cos" in feature_list
        assert "dow_sin"   in feature_list
        assert "dow_cos"   in feature_list

    def test_lag_features_exist(self, feature_list):
        lags = [f for f in feature_list if "lf_lag" in f]
        assert len(lags) >= 4, \
            f"Lag features insuffisantes : {lags}"

    def test_load_factor_distribution(self, df_ml):
        mean_lf = df_ml["load_factor"].mean()
        assert 60 < mean_lf < 95, \
            f"Load Factor moyen anormal : {mean_lf:.1f}%"