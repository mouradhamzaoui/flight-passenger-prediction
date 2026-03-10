"""
ML Training Pipeline - Delta Airlines
Modèles : Random Forest, XGBoost, LightGBM, Gradient Boosting
Tracking : MLflow complet
Standard : Airbus/Amadeus MLOps 2026
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime

from sklearn.model_selection   import train_test_split, cross_val_score, KFold
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection        import permutation_importance

import xgboost  as xgb
import lightgbm as lgb

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FEATURES_PATH = Path("data/features/delta_features_ml.csv")
MODELS_DIR    = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI    = "mlruns"
EXPERIMENT    = "delta_airlines_load_factor"
TARGET        = "load_factor"
TEST_SIZE     = 0.20
VAL_SIZE      = 0.10
RANDOM_STATE  = 42

print("=" * 65)
print("ML TRAINING PIPELINE — DELTA AIRLINES")
print("Target : Load Factor (%) | Tracker : MLflow")
print("=" * 65)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_PATH)
print(f"\n[✓] Dataset chargé : {len(df):,} lignes | {df.shape[1]} colonnes")

meta = json.load(open("data/features/feature_metadata.json"))
FEATURES = meta["features"]

X = df[FEATURES].fillna(df[FEATURES].median())
y = df[TARGET]

# ─── SPLIT ────────────────────────────────────────────────────────────────────
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE
)

print(f"[✓] Train : {len(X_train):,} | Val : {len(X_val):,} | Test : {len(X_test):,}")

# ─── MLFLOW SETUP ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)
print(f"[✓] MLflow experiment : '{EXPERIMENT}'")

# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, prefix=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true.clip(lower=1))) * 100
    return {
        f"{prefix}mae":  round(mae,  4),
        f"{prefix}rmse": round(rmse, 4),
        f"{prefix}r2":   round(r2,   4),
        f"{prefix}mape": round(mape, 4),
    }

def log_run(model, model_name, params, X_tr, y_tr, X_v, y_v, X_te, y_te, flavor="sklearn"):
    with mlflow.start_run(run_name=model_name) as run:
        # Tags
        mlflow.set_tags({
            "carrier":    "DL",
            "model_type": model_name,
            "target":     TARGET,
            "stage":      "poc",
            "engineer":   "delta-ml-platform",
        })

        # Params
        mlflow.log_params(params)
        mlflow.log_param("n_features",    len(FEATURES))
        mlflow.log_param("train_samples", len(X_tr))
        mlflow.log_param("test_samples",  len(X_te))

        # Train
        model.fit(X_tr, y_tr)

        # Predictions
        y_pred_train = model.predict(X_tr)
        y_pred_val   = model.predict(X_v)
        y_pred_test  = model.predict(X_te)

        # Metrics
        train_m = compute_metrics(y_tr, y_pred_train, "train_")
        val_m   = compute_metrics(y_v, y_pred_val,   "val_")
        test_m  = compute_metrics(y_te, y_pred_test,  "test_")

        mlflow.log_metrics({**train_m, **val_m, **test_m})

        # Cross-validation
        kf    = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_tr, y_tr,
                                     cv=kf, scoring="r2", n_jobs=-1)
        mlflow.log_metric("cv_r2_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_r2_std",  round(cv_scores.std(),  4))

        # Log model — compatible toutes versions MLflow
        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif flavor == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "feature":   FEATURES,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            fi.to_csv(f"data/models/{model_name}_feature_importance.csv", index=False)
            mlflow.log_artifact(f"data/models/{model_name}_feature_importance.csv")

        run_id = run.info.run_id
        print(f"\n  ✅ {model_name}")
        print(f"     MAE  (test) : {test_m['test_mae']:.3f}%")
        print(f"     RMSE (test) : {test_m['test_rmse']:.3f}%")
        print(f"     R²   (test) : {test_m['test_r2']:.4f}")
        print(f"     MAPE (test) : {test_m['test_mape']:.2f}%")
        print(f"     CV R² mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"     Run ID      : {run_id[:8]}...")

        return {
            "model_name": model_name,
            "run_id":     run_id,
            "model":      model,
            "y_pred":     y_pred_test,
            **test_m,
            "cv_r2_mean": round(cv_scores.mean(), 4),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 1 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Random Forest Regressor...")
rf_params = {
    "n_estimators":      300,
    "max_depth":         12,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "n_jobs":            -1,
    "random_state":      RANDOM_STATE,
}
rf_model  = RandomForestRegressor(**rf_params)
rf_result = log_run(rf_model, "RandomForest", rf_params,
                    X_train, y_train, X_val, y_val, X_test, y_test)


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 2 — XGBOOST
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] XGBoost Regressor...")
xgb_params = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbosity":         0,
}
xgb_model  = xgb.XGBRegressor(**xgb_params)
xgb_result = log_run(xgb_model, "XGBoost", xgb_params,
                     X_train, y_train, X_val, y_val, X_test, y_test,
                     flavor="xgboost")


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 3 — LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] LightGBM Regressor...")
lgb_params = {
    "n_estimators":      500,
    "max_depth":         8,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "min_child_samples": 20,
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "verbose":           -1,
}
lgb_model  = lgb.LGBMRegressor(**lgb_params)
lgb_result = log_run(lgb_model, "LightGBM", lgb_params,
                     X_train, y_train, X_val, y_val, X_test, y_test,
                     flavor="lightgbm")


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 4 — GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Gradient Boosting Regressor...")
gb_params = {
    "n_estimators":    300,
    "max_depth":       5,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "min_samples_leaf": 5,
    "random_state":    RANDOM_STATE,
}
gb_model  = GradientBoostingRegressor(**gb_params)
gb_result = log_run(gb_model, "GradientBoosting", gb_params,
                    X_train, y_train, X_val, y_val, X_test, y_test)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARAISON & SÉLECTION MEILLEUR MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("MODEL COMPARISON — DELTA AIRLINES LOAD FACTOR PREDICTION")
print("="*65)

results = [rf_result, xgb_result, lgb_result, gb_result]

comparison_df = pd.DataFrame([{
    "Model":      r["model_name"],
    "MAE (%)":    r["test_mae"],
    "RMSE (%)":   r["test_rmse"],
    "R²":         r["test_r2"],
    "MAPE (%)":   r["test_mape"],
    "CV R²":      r["cv_r2_mean"],
    "Run ID":     r["run_id"][:8],
} for r in results]).sort_values("MAE (%)")

print(comparison_df.to_string(index=False))

best = min(results, key=lambda x: x["test_mae"])
print(f"\n🏆 BEST MODEL : {best['model_name']}")
print(f"   MAE  : {best['test_mae']:.3f}% load factor")
print(f"   R²   : {best['test_r2']:.4f}")
print(f"   MAPE : {best['test_mape']:.2f}%")

# ─── SAUVEGARDE MEILLEUR MODÈLE ───────────────────────────────────────────────
with open(MODELS_DIR / "best_model.pkl", "wb") as f:
    pickle.dump(best["model"], f)

comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

# ─── SAUVEGARDE RÉSULTATS JSON ────────────────────────────────────────────────
summary = {
    "best_model":      best["model_name"],
    "best_run_id":     best["run_id"],
    "best_mae":        best["test_mae"],
    "best_r2":         best["test_r2"],
    "best_mape":       best["test_mape"],
    "trained_at":      datetime.now().isoformat(),
    "carrier":         "DL",
    "target":          TARGET,
    "n_features":      len(FEATURES),
    "train_samples":   len(X_train),
    "test_samples":    len(X_test),
    "experiment_name": EXPERIMENT,
    "all_models": [{
        "name":  r["model_name"],
        "mae":   r["test_mae"],
        "rmse":  r["test_rmse"],
        "r2":    r["test_r2"],
        "mape":  r["test_mape"],
        "run_id": r["run_id"],
    } for r in results]
}

with open(MODELS_DIR / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ─── SCALER POUR API ──────────────────────────────────────────────────────────
scaler = StandardScaler()
scaler.fit(X_train)
with open(MODELS_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(MODELS_DIR / "feature_list.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

print(f"\n[✓] Fichiers sauvegardés :")
print(f"    → data/models/best_model.pkl          ({best['model_name']})")
print(f"    → data/models/model_comparison.csv")
print(f"    → data/models/training_summary.json")
print(f"    → data/models/scaler.pkl")
print(f"    → data/models/feature_list.pkl")
print(f"\n[✓] MLflow runs : mlflow ui --port 5000")
print(f"[✓] ÉTAPE 5 COMPLÈTE — 4 modèles entraînés et trackés")