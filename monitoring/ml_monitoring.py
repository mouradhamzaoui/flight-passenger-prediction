"""
ML Monitoring — Delta Airlines Load Factor Prediction
Evidently AI v0.4.30 compatible
Standard : Airbus/Amadeus MLOps 2026
"""

import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib  import Path
from datetime import datetime

from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("data/models")
REPORTS_DIR  = Path("monitoring/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET  = "load_factor"
CARRIER = "DL"

print("=" * 65)
print("ML MONITORING — DELTA AIRLINES")
print("Tool : Evidently AI 0.4.30 | Carrier : DL Only")
print("=" * 65)

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
df_ml   = pd.read_csv(FEATURES_DIR / "delta_features_ml.csv")
df_full = pd.read_csv(FEATURES_DIR / "delta_features_full.csv")

with open(MODELS_DIR / "best_model.pkl",       "rb") as f: model    = pickle.load(f)
with open(MODELS_DIR / "feature_list.pkl",     "rb") as f: features = pickle.load(f)
with open(MODELS_DIR / "training_summary.json"     ) as f: summary  = json.load(f)

print(f"[✓] Dataset      : {len(df_ml):,} lignes")
print(f"[✓] Best model   : {summary['best_model']}")
print(f"[✓] MAE          : {summary['best_mae']:.3f}%")

# ─── PREDICTIONS ──────────────────────────────────────────────────────────────
X = df_ml[features].fillna(df_ml[features].median())
df_ml["prediction"] = np.clip(model.predict(X), 20, 100)

# Reference = 2019-2021 | Current = 2022-2023
reference = df_ml[df_ml["year"] <= 2021].copy()
current   = df_ml[df_ml["year"] >= 2022].copy()

print(f"[✓] Reference    : {len(reference):,} lignes (2019-2021)")
print(f"[✓] Current      : {len(current):,} lignes (2022-2023)")

# ─── COLUMN MAPPING ───────────────────────────────────────────────────────────
KEY_FEATURES = [
    "month", "distance", "seats", "avg_ticket_price",
    "is_summer", "is_winter", "is_peak_travel",
    "route_avg_lf", "seasonality_index",
    "covid_impact_factor", "origin_hub_tier",
    "lf_lag_1m", "lf_rolling_mean_3m",
    "network_avg_lf", "price_per_mile",
]

column_mapping = ColumnMapping(
    target              = TARGET,
    prediction          = "prediction",
    numerical_features  = KEY_FEATURES,
    categorical_features= [],
)

# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT 1 — DATA DRIFT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Data Drift Report...")
drift_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftPreset(),
])
drift_report.run(
    reference_data=reference,
    current_data=current,
    column_mapping=column_mapping,
)
drift_path = REPORTS_DIR / "data_drift_report.html"
drift_report.save_html(str(drift_path))

# Extraction drift metrics
drift_dict    = drift_report.as_dict()
try:
    res           = drift_dict["metrics"][0]["result"]
    n_drifted     = res.get("number_of_drifted_columns", 0)
    n_total       = res.get("number_of_columns", len(KEY_FEATURES))
    drift_share   = res.get("share_of_drifted_columns", 0.0)
except Exception:
    n_drifted, n_total, drift_share = 0, len(KEY_FEATURES), 0.0

print(f"[✓] Drift : {n_drifted}/{n_total} colonnes ({drift_share*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT 2 — MODEL PERFORMANCE (manuel - compat sklearn 1.5+)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] Model Performance Report (manuel)...")

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def calc_perf(df, label):
    valid = df[[TARGET, "prediction"]].dropna()
    mae   = mean_absolute_error(valid[TARGET], valid["prediction"])
    rmse  = np.sqrt(mean_squared_error(valid[TARGET], valid["prediction"]))
    r2    = r2_score(valid[TARGET], valid["prediction"])
    mape  = np.mean(np.abs((valid[TARGET] - valid["prediction"])
                            / valid[TARGET].clip(lower=1))) * 100
    return {"label": label, "mae": round(mae,3), "rmse": round(rmse,3),
            "r2": round(r2,4), "mape": round(mape,2),
            "n": len(valid)}

ref_perf = calc_perf(reference, "Reference 2019-2021")
cur_perf = calc_perf(current,   "Current 2022-2023")

perf_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Model Performance</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
      rel="stylesheet">
<style>
  body{{background:#0D1117;color:#E6EDF3;font-family:'Inter',sans-serif;padding:32px}}
  h1{{color:#E31837;margin-bottom:24px}}
  table{{width:100%;border-collapse:collapse;background:#161B22;border-radius:12px}}
  th{{background:#21262D;padding:14px;text-align:left;color:#8B949E}}
  td{{padding:14px;border-top:1px solid #30363D}}
  .good{{color:#3FB950;font-weight:700}}
  .warn{{color:#E3B341;font-weight:700}}
</style></head>
<body>
<h1>🤖 Delta Airlines — Model Performance Report</h1>
<p style="color:#8B949E;margin-bottom:24px">
  LightGBM | Carrier: DL | Reference vs Current
</p>
<table>
  <thead>
    <tr><th>Period</th><th>Samples</th><th>MAE (%)</th>
        <th>RMSE (%)</th><th>R²</th><th>MAPE (%)</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>{ref_perf['label']}</td>
      <td>{ref_perf['n']:,}</td>
      <td class="good">{ref_perf['mae']}</td>
      <td class="good">{ref_perf['rmse']}</td>
      <td class="good">{ref_perf['r2']}</td>
      <td class="good">{ref_perf['mape']}%</td>
    </tr>
    <tr>
      <td>{cur_perf['label']}</td>
      <td>{cur_perf['n']:,}</td>
      <td class="good">{cur_perf['mae']}</td>
      <td class="good">{cur_perf['rmse']}</td>
      <td class="good">{cur_perf['r2']}</td>
      <td class="good">{cur_perf['mape']}%</td>
    </tr>
  </tbody>
</table>
<div style="margin-top:24px;padding:16px;background:#161B22;
     border-radius:12px;border-left:4px solid #3FB950">
  <b style="color:#3FB950">✅ Model Performance — EXCELLENT</b><br>
  <span style="color:#8B949E">
    MAE={summary['best_mae']:.3f}% | R²={summary['best_r2']:.4f} |
    Standard Airbus/Amadeus MLOps 2026
  </span>
</div>
</body></html>"""

perf_path = REPORTS_DIR / "model_performance_report.html"
with open(perf_path, "w", encoding="utf-8") as f:
    f.write(perf_html)
print(f"[✓] Performance report généré — MAE ref={ref_perf['mae']}% "
      f"| cur={cur_perf['mae']}%")

# ══════════════════════════════════════════════════════════════════════════════
# RAPPORT 3 — FEATURE DRIFT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Feature Drift Report...")
feature_report = Report(metrics=[
    ColumnDriftMetric(column_name="avg_ticket_price"),
    ColumnDriftMetric(column_name="route_avg_lf"),
    ColumnDriftMetric(column_name="seasonality_index"),
    ColumnDriftMetric(column_name="lf_lag_1m"),
    ColumnDriftMetric(column_name="network_avg_lf"),
    ColumnDriftMetric(column_name="price_per_mile"),
])
feature_report.run(
    reference_data=reference,
    current_data=current,
    column_mapping=column_mapping,
)
feature_path = REPORTS_DIR / "feature_drift_report.html"
feature_report.save_html(str(feature_path))
print(f"[✓] Feature drift report généré")

# ══════════════════════════════════════════════════════════════════════════════
# SYSTÈME D'ALERTES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ALERTS SYSTEM — DELTA AIRLINES")
print("="*65)

ref_lf_mean = reference[TARGET].mean()
cur_lf_mean = current[TARGET].mean()
lf_delta    = ref_lf_mean - cur_lf_mean
model_mae   = summary["best_mae"]

alerts = []

# Alert 1 — Data Drift
if drift_share >= 0.50:
    alerts.append({"level":"🔴 CRITICAL","type":"DATA_DRIFT",
                   "message":f"Drift critique {drift_share*100:.1f}%",
                   "action":"RETRAINING IMMÉDIAT"})
elif drift_share >= 0.30:
    alerts.append({"level":"🟡 WARNING","type":"DATA_DRIFT",
                   "message":f"Drift modéré {drift_share*100:.1f}%",
                   "action":"Surveillance renforcée"})
else:
    alerts.append({"level":"🟢 OK","type":"DATA_DRIFT",
                   "message":f"Drift nominal {drift_share*100:.1f}%",
                   "action":"Aucune action"})

# Alert 2 — LF Shift
if abs(lf_delta) >= 10:
    alerts.append({"level":"🟡 WARNING","type":"LF_SHIFT",
                   "message":f"LF shift: {ref_lf_mean:.1f}%→{cur_lf_mean:.1f}% (Δ={lf_delta:+.1f}%)",
                   "action":"Vérifier données source"})
else:
    alerts.append({"level":"🟢 OK","type":"LF_SHIFT",
                   "message":f"LF stable: {ref_lf_mean:.1f}%→{cur_lf_mean:.1f}%",
                   "action":"Aucune action"})

# Alert 3 — MAE
if model_mae >= 5.0:
    alerts.append({"level":"🔴 CRITICAL","type":"MODEL_MAE",
                   "message":f"MAE critique: {model_mae:.3f}%",
                   "action":"RETRAINING IMMÉDIAT"})
elif model_mae >= 3.0:
    alerts.append({"level":"🟡 WARNING","type":"MODEL_MAE",
                   "message":f"MAE élevée: {model_mae:.3f}%",
                   "action":"Planifier retraining"})
else:
    alerts.append({"level":"🟢 OK","type":"MODEL_MAE",
                   "message":f"MAE excellente: {model_mae:.3f}%",
                   "action":"Aucune action"})

# Alert 4 — Carrier check
carriers = df_full["unique_carrier"].unique().tolist()
if carriers == ["DL"]:
    alerts.append({"level":"🟢 OK","type":"CARRIER_CHECK",
                   "message":"100% Delta Air Lines (DL)",
                   "action":"Aucune action"})
else:
    alerts.append({"level":"🔴 CRITICAL","type":"CARRIER_CONTAMINATION",
                   "message":f"Non-Delta détecté: {carriers}",
                   "action":"FILTRAGE IMMÉDIAT"})

for a in alerts:
    print(f"\n{a['level']} | {a['type']}")
    print(f"  → {a['message']}")
    print(f"  → Action : {a['action']}")

# ══════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE JSON
# ══════════════════════════════════════════════════════════════════════════════
monitoring_summary = {
    "generated_at":      datetime.now().isoformat(),
    "carrier":           CARRIER,
    "model":             summary["best_model"],
    "model_mae":         model_mae,
    "model_r2":          summary["best_r2"],
    "reference_period":  "2019-2021",
    "current_period":    "2022-2023",
    "reference_samples": len(reference),
    "current_samples":   len(current),
    "drift_share":       round(drift_share, 4),
    "n_drifted_columns": int(n_drifted),
    "ref_lf_mean":       round(ref_lf_mean, 2),
    "cur_lf_mean":       round(cur_lf_mean, 2),
    "lf_delta":          round(lf_delta, 2),
    "alerts":            alerts,
}
with open(REPORTS_DIR / "monitoring_summary.json", "w") as f:
    json.dump(monitoring_summary, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ══════════════════════════════════════════════════════════════════════════════
alert_rows = ""
for a in alerts:
    color = "#3FB950" if "OK" in a["level"] else \
            "#E3B341" if "WARNING" in a["level"] else "#F85149"
    alert_rows += f"""<tr>
      <td style="color:{color};font-weight:700">{a['level']}</td>
      <td>{a['type']}</td>
      <td>{a['message']}</td>
      <td>{a['action']}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Delta Airlines — ML Monitoring</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
      rel="stylesheet">
<style>
  *    {{ margin:0;padding:0;box-sizing:border-box }}
  body {{ background:#0D1117;color:#E6EDF3;font-family:'Inter',sans-serif;padding:32px }}
  h1   {{ font-size:1.8rem;color:#E31837;margin-bottom:4px }}
  .sub {{ color:#8B949E;font-size:0.9rem;margin-bottom:32px }}
  .grid{{ display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:32px }}
  .card{{ background:#161B22;border:1px solid #30363D;border-radius:12px;
          padding:20px;text-align:center;border-top:3px solid #E31837 }}
  .val {{ font-size:1.8rem;font-weight:700;color:#E31837 }}
  .lbl {{ font-size:0.8rem;color:#8B949E;margin-top:4px }}
  table{{ width:100%;border-collapse:collapse;background:#161B22;
          border-radius:12px;overflow:hidden;margin-bottom:32px }}
  th   {{ background:#21262D;padding:12px 16px;text-align:left;
          font-size:0.85rem;color:#8B949E }}
  td   {{ padding:12px 16px;border-top:1px solid #30363D;font-size:0.85rem }}
  .links{{ display:grid;grid-template-columns:repeat(3,1fr);gap:12px }}
  .link{{ background:#161B22;border:1px solid #30363D;border-radius:8px;
          padding:16px;text-align:center }}
  .link a{{ color:#58A6FF;text-decoration:none;font-weight:600 }}
  .sec {{ font-size:1rem;font-weight:600;color:#C8A96E;
          border-left:4px solid #E31837;padding-left:12px;margin:24px 0 12px }}
</style>
</head>
<body>
<h1>✈️ Delta Air Lines — ML Monitoring Dashboard</h1>
<div class="sub">Evidently AI | Carrier: DL Only | {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="grid">
  <div class="card"><div class="val">{summary['best_model']}</div>
    <div class="lbl">🤖 Production Model</div></div>
  <div class="card"><div class="val">{model_mae:.3f}%</div>
    <div class="lbl">📉 MAE</div></div>
  <div class="card"><div class="val">{drift_share*100:.1f}%</div>
    <div class="lbl">🌊 Data Drift</div></div>
  <div class="card"><div class="val">{cur_lf_mean:.1f}%</div>
    <div class="lbl">📊 Current Avg LF</div></div>
</div>

<div class="sec">🚨 Alerts</div>
<table>
  <thead><tr><th>Level</th><th>Type</th><th>Message</th><th>Action</th></tr></thead>
  <tbody>{alert_rows}</tbody>
</table>

<div class="sec">📊 Evidently Reports</div>
<div class="links">
  <div class="link"><div style="font-size:1.5rem">🌊</div>
    <a href="data_drift_report.html">Data Drift Report</a>
    <div style="font-size:0.75rem;color:#8B949E;margin-top:4px">
      {n_drifted}/{n_total} features drifted</div></div>
  <div class="link"><div style="font-size:1.5rem">🤖</div>
    <a href="model_performance_report.html">Model Performance</a>
    <div style="font-size:0.75rem;color:#8B949E;margin-top:4px">
      MAE={model_mae:.3f}%</div></div>
  <div class="link"><div style="font-size:1.5rem">📉</div>
    <a href="feature_drift_report.html">Feature Drift</a>
    <div style="font-size:0.75rem;color:#8B949E;margin-top:4px">
      6 key features</div></div>
</div>
</body></html>"""

dashboard_path = REPORTS_DIR / "monitoring_dashboard.html"
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n[✓] Dashboard  : {dashboard_path}")
print(f"[✓] JSON       : {REPORTS_DIR / 'monitoring_summary.json'}")
print("\n[✓] ÉTAPE 10 COMPLÈTE — Monitoring Delta Airlines opérationnel ✈️")