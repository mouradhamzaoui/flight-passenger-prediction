"""
POC Streamlit - Delta Airlines Load Factor Prediction
Dashboard interactif - Standard Airbus/Amadeus MLOps 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delta Airlines — ML Load Factor Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME DELTA ──────────────────────────────────────────────────────────────
DELTA_RED  = "#E31837"
DELTA_BLUE = "#003057"
DELTA_GOLD = "#C8A96E"
BG_DARK    = "#0D1117"
BG_CARD    = "#161B22"
TEXT       = "#E6EDF3"

st.markdown(f"""
<style>
  /* Global */
  .stApp {{ background-color: {BG_DARK}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color: {BG_CARD}; }}

  /* Header */
  .delta-header {{
    background: linear-gradient(135deg, {DELTA_BLUE} 0%, {DELTA_RED} 100%);
    padding: 24px 32px; border-radius: 12px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
  }}
  .delta-header h1 {{ font-size: 1.8rem; font-weight: 700; color: white; margin: 0; }}
  .delta-header p  {{ color: rgba(255,255,255,0.85); margin: 4px 0 0; font-size: 0.95rem; }}

  /* KPI Cards */
  .kpi-card {{
    background: {BG_CARD}; border: 1px solid #30363D;
    border-radius: 12px; padding: 20px; text-align: center;
    border-top: 3px solid {DELTA_RED};
  }}
  .kpi-value {{ font-size: 2rem; font-weight: 700; color: {DELTA_RED}; }}
  .kpi-label {{ font-size: 0.85rem; color: #8B949E; margin-top: 4px; }}

  /* Prediction box */
  .pred-box {{
    background: linear-gradient(135deg, {DELTA_BLUE}22, {DELTA_RED}22);
    border: 2px solid {DELTA_RED}; border-radius: 16px;
    padding: 32px; text-align: center; margin: 16px 0;
  }}
  .pred-value {{ font-size: 3.5rem; font-weight: 800; color: {DELTA_RED}; }}
  .pred-label {{ font-size: 1rem; color: {TEXT}; opacity: 0.8; margin-top: 8px; }}

  /* Gauge label */
  .gauge-good  {{ color: #3FB950; font-weight: 600; }}
  .gauge-avg   {{ color: {DELTA_GOLD}; font-weight: 600; }}
  .gauge-low   {{ color: {DELTA_RED}; font-weight: 600; }}

  /* Hide streamlit default elements */
  #MainMenu {{ visibility: hidden; }}
  footer     {{ visibility: hidden; }}
  header     {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    models_dir   = Path("data/models")
    features_dir = Path("data/features")

    with open(models_dir / "best_model.pkl",   "rb") as f:
        model = pickle.load(f)
    with open(models_dir / "feature_list.pkl", "rb") as f:
        features = pickle.load(f)
    with open(models_dir / "scaler.pkl",       "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "training_summary.json") as f:
        summary = json.load(f)
    with open(features_dir / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    df_full = pd.read_csv(features_dir / "delta_features_full.csv")
    df_ml   = pd.read_csv(features_dir / "delta_features_ml.csv")

    return model, features, scaler, summary, encoders, df_full, df_ml

model, features, scaler, summary, encoders, df_full, df_ml = load_artifacts()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding:16px 0;">
      <div style="font-size:2.5rem;">✈️</div>
      <div style="font-size:1.1rem; font-weight:700; color:{DELTA_RED};">Delta Air Lines</div>
      <div style="font-size:0.8rem; color:#8B949E;">ML Prediction Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "🎯 Predict Flight",
        "📊 Route Analysis",
        "🗺️  Network Map",
        "🤖 Model Performance",
    ])
    st.divider()

    st.markdown(f"""
    <div style="font-size:0.8rem; color:#8B949E; padding:8px;">
      <b style="color:{DELTA_GOLD};">Best Model</b><br>
      {summary['best_model']}<br><br>
      <b style="color:{DELTA_GOLD};">MAE</b> {summary['best_mae']:.2f}%<br>
      <b style="color:{DELTA_GOLD};">R²</b>  {summary['best_r2']:.4f}<br>
      <b style="color:{DELTA_GOLD};">MAPE</b> {summary['best_mape']:.2f}%<br><br>
      <b style="color:{DELTA_GOLD};">Carrier</b> DL — Delta Only<br>
      <b style="color:{DELTA_GOLD};">Features</b> {summary['n_features']}<br>
      <b style="color:{DELTA_GOLD};">Train samples</b> {summary['train_samples']:,}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown(f"""
    <div class="delta-header">
      <div>✈️</div>
      <div>
        <h1>Delta Air Lines — Load Factor Prediction Platform</h1>
        <p>ML-Powered Flight Demand Forecasting | Standard Airbus/Amadeus MLOps 2026 | Carrier: DL Only</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    avg_lf   = df_full["load_factor"].mean()
    total_pax = df_full["passengers"].sum()
    n_routes  = df_full["route"].nunique() if "route" in df_full.columns else 20
    best_year_lf = df_full.groupby("year")["load_factor"].mean().max()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{avg_lf:.1f}%</div>
          <div class="kpi-label">📊 Avg Load Factor</div></div>""",
          unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{total_pax/1e6:.1f}M</div>
          <div class="kpi-label">👥 Total Passengers</div></div>""",
          unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{n_routes}</div>
          <div class="kpi-label">🛫 Routes Analyzed</div></div>""",
          unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{summary['best_model']}</div>
          <div class="kpi-label">🤖 Best ML Model</div></div>""",
          unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Timeline + Heatmap
    col1, col2 = st.columns([3, 2])

    with col1:
        monthly = df_full.groupby(["year","month"]).agg(
            avg_lf=("load_factor","mean"),
            total_pax=("passengers","sum")
        ).reset_index()
        monthly["date"] = pd.to_datetime(
            monthly["year"].astype(str)+"-"+monthly["month"].astype(str).str.zfill(2)
        )
        monthly = monthly.sort_values("date")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=monthly["date"], y=monthly["avg_lf"],
            name="Load Factor %", line=dict(color=DELTA_RED, width=2),
            fill="tozeroy", fillcolor="rgba(227,24,55,0.1)"
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=monthly["date"], y=monthly["total_pax"],
            name="Passengers", marker_color=DELTA_BLUE, opacity=0.35
        ), secondary_y=True)
        fig.add_vrect(x0="2020-03-01", x1="2021-07-01",
                      fillcolor="rgba(255,123,114,0.1)",
                      annotation_text="COVID-19", line_width=0)
        fig.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=320,
            title=dict(text="📈 Monthly Load Factor & Passengers (2019-2023)",
                       font=dict(size=14)),
            legend=dict(bgcolor=BG_CARD),
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pivot = df_full.pivot_table(
            values="load_factor", index="year", columns="month", aggfunc="mean"
        ).round(1)
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig2 = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0,DELTA_BLUE],[0.5,DELTA_GOLD],[1,DELTA_RED]],
            text=pivot.values.round(0), texttemplate="%{text}%",
            textfont=dict(size=9),
        ))
        fig2.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=320,
            title=dict(text="🌡️ Load Factor Heatmap", font=dict(size=14)),
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT FLIGHT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Flight":
    st.markdown(f"## 🎯 Predict Load Factor — Delta Air Lines")

    ROUTES = [
        ("ATL","LGA",762), ("ATL","BOS",1099), ("ATL","LAX",1946),
        ("ATL","MCO",403), ("ATL","MIA",662),  ("ATL","DTW",594),
        ("ATL","MSP",907), ("ATL","SLC",1589), ("ATL","SEA",2182),
        ("ATL","JFK",760), ("DTW","MSP",528),  ("DTW","BOS",632),
        ("DTW","LGA",502), ("MSP","SLC",987),  ("MSP","SEA",1399),
        ("SLC","SEA",689), ("SEA","LAX",954),  ("SEA","JFK",2422),
        ("LGA","MCO",1074),("BOS","MCO",1123),
    ]
    ROUTE_LABELS = {f"{o}→{d}": (o,d,dist) for o,d,dist in ROUTES}

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### ✈️ Flight Parameters")
        route_sel  = st.selectbox("Route",  list(ROUTE_LABELS.keys()))
        origin, dest, distance = ROUTE_LABELS[route_sel]
        month_sel  = st.slider("Month",       1, 12, 7)
        dow_sel    = st.slider("Day of Week", 1, 7,  5,
                                help="1=Mon … 7=Sun")
        year_sel   = st.selectbox("Year", [2023, 2024, 2025])
        seats_sel  = st.slider("Seats", 100, 250, 180)
        price_sel  = st.slider("Avg Ticket Price ($)", 80, 600, 220)
        weather_sel = st.selectbox("Weather",
                                   ["CLEAR","CLOUDY","RAIN","SNOW"])
        holiday_sel = st.checkbox("Holiday Period", value=False)

    with col_right:
        st.markdown("### 📊 Prediction")

        weather_map = {"CLEAR": 0, "CLOUDY": 1, "RAIN": 2, "SNOW": 3}
        season_map  = {12:0.98, 1:0.88, 2:0.85, 3:0.95, 4:0.97,
                       5:1.00, 6:1.08, 7:1.12, 8:1.10,
                       9:0.92, 10:0.94, 11:1.02}

        # Calcul moyennes historiques pour la route sélectionnée
        route_key = f"{origin}_{dest}"
        route_data = df_ml[
            (df_ml.get("origin_encoded", pd.Series(dtype=float)).notna())
        ] if "origin_encoded" in df_ml.columns else df_ml

        # Médianes features route depuis dataset
        route_avg_lf       = df_full[
            (df_full["origin"]==origin) & (df_full["dest"]==dest)
        ]["load_factor"].mean() if len(df_full[
            (df_full["origin"]==origin) & (df_full["dest"]==dest)
        ]) > 0 else df_full["load_factor"].mean()

        route_std_lf       = df_full[
            (df_full["origin"]==origin) & (df_full["dest"]==dest)
        ]["load_factor"].std() if len(df_full[
            (df_full["origin"]==origin) & (df_full["dest"]==dest)
        ]) > 0 else df_full["load_factor"].std()

        # Route popularity rank
        route_pop = df_full.groupby("route")["passengers"].sum()
        route_pop_rank = int(route_pop.rank(ascending=False).get(
            f"{origin}_{dest}", 10))

        HUB_TIER = {"ATL":1,"DTW":1,"MSP":1,"SLC":2,
                    "SEA":2,"BOS":2,"LGA":2,"LAX":2,
                    "JFK":2,"MCO":3,"MIA":3}

        month_sin = np.sin(2 * np.pi * month_sel / 12)
        month_cos = np.cos(2 * np.pi * month_sel / 12)
        dow_sin   = np.sin(2 * np.pi * dow_sel / 7)
        dow_cos   = np.cos(2 * np.pi * dow_sel / 7)

        # Encodage route
        try:
            r_enc = encoders["route"].transform([f"{origin}_{dest}"])[0]
            o_enc = encoders["origin"].transform([origin])[0]
            d_enc = encoders["dest"].transform([dest])[0]
        except:
            r_enc, o_enc, d_enc = 0, 0, 0

        # Network stats
        net_avg_lf    = df_full[df_full["month"]==month_sel]["load_factor"].mean()
        net_avg_price = df_full[df_full["month"]==month_sel]["avg_ticket_price"].mean()

        # Valeurs moyennes lag depuis dataset
        median_vals = df_ml.median()

        input_data = {
            "year": year_sel, "month": month_sel,
            "day_of_week": dow_sel, "quarter": (month_sel-1)//3+1,
            "month_sin": month_sin, "month_cos": month_cos,
            "dow_sin": dow_sin, "dow_cos": dow_cos,
            "is_weekend": int(dow_sel >= 6),
            "is_monday": int(dow_sel == 1),
            "is_friday": int(dow_sel == 5),
            "is_summer": int(month_sel in [6,7,8]),
            "is_winter": int(month_sel in [12,1,2]),
            "is_spring": int(month_sel in [3,4,5]),
            "is_fall":   int(month_sel in [9,10,11]),
            "is_peak_travel": int(month_sel in [6,7,8,11,12]),
            "is_holiday_period": int(holiday_sel),
            "is_thanksgiving": int(month_sel==11 and holiday_sel),
            "is_xmas_newyear": int(month_sel==12 and holiday_sel),
            "is_july4": int(month_sel==7 and holiday_sel),
            "distance": distance, "seats": seats_sel,
            "avg_ticket_price": price_sel,
            "price_per_mile": price_sel / max(distance, 1),
            "revenue_per_seat": (seats_sel * 0.85 * price_sel) / max(seats_sel, 1),
            "yield_metric": price_sel / max(distance, 1) * 100,
            "is_long_haul":   int(distance >= 1500),
            "is_medium_haul": int(500 <= distance < 1500),
            "is_short_haul":  int(distance < 500),
            "weather_encoded": weather_map[weather_sel],
            "route_avg_lf":        route_avg_lf,
            "route_std_lf":        route_std_lf if not np.isnan(route_std_lf) else 8.0,
            "route_max_lf":        df_full["load_factor"].max(),
            "route_avg_price":     price_sel,
            "route_avg_distance":  distance,
            "route_lf_cv":         0.10,
            "route_popularity_rank": route_pop_rank,
            "route_encoded":       r_enc,
            "origin_hub_tier":     HUB_TIER.get(origin, 3),
            "dest_hub_tier":       HUB_TIER.get(dest, 3),
            "hub_to_hub":          int(HUB_TIER.get(origin,3)==1 and HUB_TIER.get(dest,3)==1),
            "hub_to_spoke":        int(HUB_TIER.get(origin,3)==1 and HUB_TIER.get(dest,3)>=2),
            "is_atl_flight":       int(origin=="ATL" or dest=="ATL"),
            "origin_avg_lf":       df_full[df_full["origin"]==origin]["load_factor"].mean()
                                   if len(df_full[df_full["origin"]==origin])>0
                                   else df_full["load_factor"].mean(),
            "origin_n_routes":     df_full[df_full["origin"]==origin]["dest"].nunique()
                                   if len(df_full[df_full["origin"]==origin])>0 else 5,
            "origin_encoded":      o_enc,
            "dest_encoded":        d_enc,
            "network_avg_lf":      net_avg_lf,
            "network_avg_price":   net_avg_price,
            "price_vs_network_avg": price_sel - net_avg_price,
            "is_post_covid":       int(year_sel >= 2022),
            "is_covid_period":     0,
            "recovery_phase":      0,
            "covid_impact_factor": 1.0,
            "seasonality_index":   season_map.get(month_sel, 1.0),
            "lf_lag_1m":           median_vals.get("lf_lag_1m",  route_avg_lf),
            "lf_lag_2m":           median_vals.get("lf_lag_2m",  route_avg_lf),
            "lf_lag_3m":           median_vals.get("lf_lag_3m",  route_avg_lf),
            "lf_lag_6m":           median_vals.get("lf_lag_6m",  route_avg_lf),
            "lf_lag_12m":          median_vals.get("lf_lag_12m", route_avg_lf),
            "lf_rolling_mean_3m":  median_vals.get("lf_rolling_mean_3m",  route_avg_lf),
            "lf_rolling_mean_6m":  median_vals.get("lf_rolling_mean_6m",  route_avg_lf),
            "lf_rolling_mean_12m": median_vals.get("lf_rolling_mean_12m", route_avg_lf),
            "lf_rolling_std_3m":   median_vals.get("lf_rolling_std_3m",  8.0),
            "lf_rolling_std_6m":   median_vals.get("lf_rolling_std_6m",  8.0),
            "lf_mom_change":       0.0,
            "lf_yoy_change":       0.0,
        }

        X_input = pd.DataFrame([input_data])[features]
        X_input = X_input.fillna(X_input.median())
        pred_lf = float(model.predict(X_input)[0])
        pred_lf = np.clip(pred_lf, 20, 100)

        # Couleur selon seuil
        if pred_lf >= 85:
            color, label = "#3FB950", "🟢 Excellent"
        elif pred_lf >= 70:
            color, label = DELTA_GOLD, "🟡 Good"
        else:
            color, label = DELTA_RED, "🔴 Low"

        st.markdown(f"""
        <div class="pred-box">
          <div style="font-size:0.9rem; color:#8B949E; margin-bottom:8px;">
            ✈️ {origin} → {dest} | Month {month_sel} | {seats_sel} seats
          </div>
          <div class="pred-value" style="color:{color};">{pred_lf:.1f}%</div>
          <div class="pred-label">Predicted Load Factor</div>
          <div style="margin-top:12px; font-size:1.1rem;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_lf,
            delta={"reference": route_avg_lf,
                   "valueformat": ".1f",
                   "suffix": "% vs route avg"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT},
                "bar":  {"color": color},
                "bgcolor": BG_CARD,
                "steps": [
                    {"range": [0,  60], "color": "rgba(227,24,55,0.2)"},
                    {"range": [60, 80], "color": "rgba(200,169,110,0.2)"},
                    {"range": [80,100], "color": "rgba(63,185,80,0.2)"},
                ],
                "threshold": {
                    "line": {"color": DELTA_GOLD, "width": 3},
                    "value": route_avg_lf
                }
            },
            number={"suffix": "%", "font": {"color": color, "size": 40}},
            title={"text": "Load Factor Gauge",
                   "font": {"color": TEXT, "size": 14}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor=BG_CARD, font=dict(color=TEXT),
            height=260, margin=dict(t=30, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Passengers estimate
        pax_est = int(seats_sel * pred_lf / 100)
        rev_est = pax_est * price_sel
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:12px;">
          <div class="kpi-card">
            <div class="kpi-value" style="font-size:1.6rem;">{pax_est}</div>
            <div class="kpi-label">👥 Est. Passengers</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-value" style="font-size:1.6rem;">${rev_est:,.0f}</div>
            <div class="kpi-label">💰 Est. Revenue</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ROUTE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Route Analysis":
    st.markdown("## 📊 Route Performance Analysis")

    if "route" not in df_full.columns:
        df_full["route"] = df_full["origin"] + "→" + df_full["dest"]

    route_stats = df_full.groupby("route").agg(
        avg_lf=("load_factor","mean"),
        std_lf=("load_factor","std"),
        total_pax=("passengers","sum"),
        avg_price=("avg_ticket_price","mean"),
        distance=("distance","first"),
    ).reset_index().sort_values("avg_lf", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            route_stats, x="avg_lf", y="route",
            orientation="h",
            color="avg_lf",
            color_continuous_scale=[[0,DELTA_BLUE],[0.5,DELTA_GOLD],[1,DELTA_RED]],
            labels={"avg_lf":"Avg Load Factor (%)","route":"Route"},
            title="📊 Average Load Factor by Route"
        )
        fig.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=550,
            margin=dict(t=40,b=20,l=20,r=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            route_stats, x="distance", y="avg_lf",
            size="total_pax", color="avg_price",
            hover_name="route",
            color_continuous_scale=[[0,DELTA_BLUE],[1,DELTA_RED]],
            labels={"distance":"Distance (miles)",
                    "avg_lf":"Avg Load Factor (%)",
                    "avg_price":"Avg Price ($)"},
            title="🔵 Distance vs Load Factor (size=passengers)"
        )
        fig2.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=550,
            margin=dict(t=40,b=20,l=20,r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Monthly trend by route
    st.markdown("### 📈 Monthly Load Factor — Route Deep Dive")
    sel_route = st.selectbox("Select Route", route_stats["route"].tolist())
    origin_r, dest_r = sel_route.split("_") if "_" in sel_route else sel_route.split("→")

    route_monthly = df_full[
        (df_full["origin"]==origin_r.strip()) &
        (df_full["dest"]==dest_r.strip())
    ].groupby(["year","month"]).agg(
        avg_lf=("load_factor","mean"),
        total_pax=("passengers","sum")
    ).reset_index()

    if len(route_monthly) > 0:
        route_monthly["date"] = pd.to_datetime(
            route_monthly["year"].astype(str)+"-"+
            route_monthly["month"].astype(str).str.zfill(2)
        )
        fig3 = px.line(
            route_monthly.sort_values("date"),
            x="date", y="avg_lf",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=[DELTA_RED],
            title=f"📈 Load Factor Trend — {sel_route}"
        )
        fig3.add_hline(y=route_monthly["avg_lf"].mean(),
                       line_dash="dash", line_color=DELTA_GOLD,
                       annotation_text=f"Route avg: {route_monthly['avg_lf'].mean():.1f}%")
        fig3.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=350,
            margin=dict(t=40,b=20,l=20,r=20)
        )
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — NETWORK MAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️  Network Map":
    st.markdown("## 🗺️ Delta Air Lines — Hub Network Map USA")

    COORDS = {
        "ATL":(33.64,-84.43), "DTW":(42.21,-83.35),
        "MSP":(44.88,-93.22), "SLC":(40.79,-111.98),
        "SEA":(47.45,-122.31), "BOS":(42.36,-71.01),
        "LGA":(40.78,-73.87),  "LAX":(33.94,-118.41),
        "JFK":(40.64,-73.78),  "MCO":(28.43,-81.31),
        "MIA":(25.80,-80.29),
    }
    ROUTES = [
        ("ATL","LGA"),("ATL","BOS"),("ATL","LAX"),("ATL","MCO"),
        ("ATL","MIA"),("ATL","DTW"),("ATL","MSP"),("ATL","SLC"),
        ("ATL","SEA"),("ATL","JFK"),("DTW","MSP"),("DTW","BOS"),
        ("DTW","LGA"),("MSP","SLC"),("MSP","SEA"),("SLC","SEA"),
        ("SEA","LAX"),("SEA","JFK"),("LGA","MCO"),("BOS","MCO"),
    ]
    hub_stats = df_full.groupby("origin").agg(
        avg_lf=("load_factor","mean"),
        total_pax=("passengers","sum"),
    ).reset_index()
    hub_dict = {r["origin"]: r for _, r in hub_stats.iterrows()}

    fig = go.Figure()

    # Routes lines
    for o, d in ROUTES:
        if o in COORDS and d in COORDS:
            lf_val = (hub_dict.get(o,{}).get("avg_lf", 80) +
                      hub_dict.get(d,{}).get("avg_lf", 80)) / 2
            fig.add_trace(go.Scattergeo(
                lat=[COORDS[o][0], COORDS[d][0]],
                lon=[COORDS[o][1], COORDS[d][1]],
                mode="lines",
                line=dict(width=1.5,
                          color=f"rgba(227,24,55,{min(lf_val/100*1.2, 0.7):.2f})"),
                showlegend=False, hoverinfo="skip"
            ))

    # Airport markers
    for ap, (lat, lon) in COORDS.items():
        stats = hub_dict.get(ap, {})
        avg_lf    = stats.get("avg_lf",    80)
        total_pax = stats.get("total_pax", 0)
        size = 15 + (total_pax / max(h.get("total_pax",1)
                                      for h in hub_dict.values()) * 35)
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon], text=[ap],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=size, color=avg_lf,
                        colorscale=[[0,DELTA_BLUE],[0.5,DELTA_GOLD],[1,DELTA_RED]],
                        cmin=70, cmax=95,
                        colorbar=dict(title="Avg LF%", x=1.02),
                        line=dict(color="white", width=1.5)),
            name=ap,
            hovertemplate=(f"<b>{ap}</b><br>"
                           f"Avg LF: {avg_lf:.1f}%<br>"
                           f"Total Pax: {int(total_pax):,}<extra></extra>")
        ))

    fig.update_geos(
        scope="usa", bgcolor=BG_DARK,
        landcolor="#1C2128", coastlinecolor="#30363D",
        showlakes=True, lakecolor=BG_DARK,
        showsubunits=True, subunitcolor="#30363D"
    )
    fig.update_layout(
        paper_bgcolor=BG_DARK, font=dict(color=TEXT),
        height=580, showlegend=False,
        title=dict(text="🗺️ Delta Air Lines Network — Load Factor by Hub",
                   font=dict(size=16, color=TEXT), x=0.5),
        margin=dict(t=50, b=10, l=10, r=10),
        geo=dict(bgcolor=BG_DARK)
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("## 🤖 Model Performance Comparison")

    comp_df = pd.DataFrame(summary["all_models"])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            comp_df.sort_values("mae"),
            x="name", y="mae",
            color="mae",
            color_continuous_scale=[[0,"#3FB950"],[0.5,DELTA_GOLD],[1,DELTA_RED]],
            title="📉 MAE by Model (lower = better)",
            labels={"mae":"MAE (%)", "name":"Model"}
        )
        fig.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=350,
            coloraxis_showscale=False,
            margin=dict(t=40,b=20,l=20,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            comp_df.sort_values("r2", ascending=False),
            x="name", y="r2",
            color="r2",
            color_continuous_scale=[[0,DELTA_RED],[0.5,DELTA_GOLD],[1,"#3FB950"]],
            title="📈 R² Score by Model (higher = better)",
            labels={"r2":"R² Score", "name":"Model"}
        )
        fig2.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=350,
            coloraxis_showscale=False,
            margin=dict(t=40,b=20,l=20,r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    best_fi_path = Path(f"data/models/{summary['best_model']}_feature_importance.csv")
    if best_fi_path.exists():
        st.markdown(f"### 🔍 Feature Importance — {summary['best_model']}")
        fi_df = pd.read_csv(best_fi_path).head(20)
        fig3 = px.bar(
            fi_df.sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=[[0,DELTA_BLUE],[0.5,DELTA_GOLD],[1,DELTA_RED]],
            title=f"Top 20 Features — {summary['best_model']}",
        )
        fig3.update_layout(
            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TEXT), height=550,
            coloraxis_showscale=False,
            margin=dict(t=40,b=20,l=20,r=20)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    st.markdown("### 📋 Full Metrics Table")
    st.dataframe(
        comp_df[["name","mae","rmse","r2","mape"]].sort_values("mae"),
        use_container_width=True,
        hide_index=True
    )