import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CarbonAI Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background-color: #f6fbf8;
}

/* GLOBAL TEXT FIX (IMPORTANT) */
html, body, [class*="css"] {
    color: #064e3b !important;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #064e3b !important;
}

/* HERO CARD */
.hero-card {
    background: linear-gradient(135deg, #065f46, #16a34a);
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    color: #064e3b;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* METRIC CARDS */
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #d1fae5;
    color: #991b1b;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
}

/* AI RECOMMENDATION CARD */
.recommend-card {
    background: #ecfdf5;
    border: 1px solid #4ade80;
    padding: 1.2rem;
    border-radius: 16px;
    color: #065b46;
    font-weight: 500;
}

/* INSIGHT BOX */
.insight-card {
    background: #f0fdf4;
    border-left: 6px solid #22c55e;
    padding: 1rem;
    border-radius: 12px;
    color: #064e3b;
}

/* RISK BADGES */
.risk-low {
    background: #dcfce7;
    color: #166534;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    font-weight: bold;
}

.risk-medium {
    background: #fef9c3;
    color: #854d0e;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    font-weight: bold;
}

.risk-high {
    background: #fee2e2;
    color: #991b1b;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    font-weight: bold;
}

/* SECTION TITLES */
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #064e3b;
    margin-top: 1rem;
}

/* METRIC COMPONENT FIX */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #d1fae5;
    padding: 0.8rem;
    border-radius: 16px;
    color: #90EE90 !important;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] {
    color: #90EE90 !important;
}

/* BUTTON */
.stButton button {
    background-color: #16a34a;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #15803d;
}

/* =========================================================
   TEXT VISIBILITY FIXES FOR CARDS
   ========================================================= */

/* Metric card labels */
div[data-testid="stMetricLabel"] p {
    color: #1e293b  !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Metric card values */
div[data-testid="stMetricValue"] {
    color: #1e293b !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* Metric delta text */
div[data-testid="stMetricDelta"] {
    color: #16a34a !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Force text inside custom cards */
.metric-card, .metric-card * {
    color: #475569 !important;
    opacity: 1 !important;
}

.insight-card, .insight-card * {
    color: #064e3b !important;
    opacity: 1 !important;
}

.recommend-card, .recommend-card * {
    color: #065f46 !important;
    opacity: 1 !important;
}

/* Keep hero subtitle readable */
.hero-card p {
    color: #ecfdf5 !important;
    opacity: 1 !important;
}

/* Keep hero title readable */
.hero-card h1 {
    color: #ecfdf5 !important;
    opacity: 1 !important;
}

/* Section titles */
.section-title {
    color: #065f46 !important;
    opacity: 1 !important;
}

/* CHANGE METRIC HEADING COLOR */
div[data-testid="stMetricLabel"] {
    color: #16a34a !important;   /* change this color */
    font-weight: 600 !important;
}

/* Optional: make it slightly darker */
div[data-testid="stMetricLabel"] p {
    color: #065f46 !important;
}


/* Metric headings (Current Emissions, etc.) */
div[data-testid="stMetricLabel"] {
    color: #6b7280 !important;   /* GREY */ 
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Inner text */
div[data-testid="stMetricLabel"] p {
    color: #6b7280 !important;   /* ✅ GREY */
}
    opacity: 1 !important;
}

/* Metric values (numbers) */
div[data-testid="stMetricLabel"] p {
    color: #6b7280 !important;   /* ✅ GREY */
}
    font-weight: 700 !important;
    opacity: 1 !important;
}

/* Delta */
div[data-testid="stMetricLabel"] p {
    color: #6b7280 !important;   /* ✅ GREY */
}
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* REMOVE FADED EFFECT */
div[data-testid="stMetricLabel"] p {
    color: #6b7280 !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    country_model = None
    industry_model = None

    if os.path.exists("country_co2_model.joblib"):
        country_model = joblib.load("country_co2_model.joblib")

    if os.path.exists("carbon_emission_model.joblib"):
        industry_model = joblib.load("carbon_emission_model.joblib")

    return country_model, industry_model


country_model, industry_model = load_models()

# =========================================================
# HELPERS
# =========================================================
def calculate_policy_reduction(ev, renewable, efficiency, carbon_tax, coal_to_solar):
    """
    Dummy policy logic for UI simulation.
    You can replace this with real model logic later.
    """
    reduction = (
        ev * 0.10 +
        renewable * 0.28 +
        efficiency * 0.18 +
        carbon_tax * 0.12 +
        coal_to_solar * 0.22
    ) / 100.0

    reduction = min(max(reduction, 0.03), 0.65)
    return reduction


def get_risk_level(predicted_2030):
    if predicted_2030 < 250:
        return "Low", "risk-low"
    elif predicted_2030 < 500:
        return "Medium", "risk-medium"
    return "High", "risk-high"


def build_forecast_series(current_emission, reduction_factor, start_year, end_year):
    years_hist = list(range(start_year, 2024))
    if len(years_hist) == 0:
        years_hist = [2020, 2021, 2022, 2023]

    # historical synthetic series
    hist_vals = []
    base_seed = current_emission * 0.82
    step = (current_emission - base_seed) / max(len(years_hist) - 1, 1)

    for i, _ in enumerate(years_hist):
        val = base_seed + i * step + np.sin(i) * 5
        hist_vals.append(max(val, 0))

    years_forecast = list(range(2024, end_year + 1))
    baseline_vals = []
    scenario_vals = []

    base = current_emission
    scenario = current_emission

    for i, y in enumerate(years_forecast, start=1):
        baseline_growth = 1.015
        scenario_growth = 1.015 - (reduction_factor * 0.10)

        base = base * baseline_growth
        scenario = scenario * max(scenario_growth, 0.90)

        baseline_vals.append(base)
        scenario_vals.append(scenario)

    return years_hist, hist_vals, years_forecast, baseline_vals, scenario_vals


def predict_country_emission(region, end_year, reduction_factor):
    """
    Uses country model if available, else simulated baseline.
    """
    # base feature values for country model
    year = end_year
    lag_1 = 420.0
    lag_2 = 405.0
    lag_3 = 390.0
    rolling_mean_3 = np.mean([lag_1, lag_2, lag_3])

    region_factor_map = {
        "India": 1.35,
        "United States": 1.25,
        "China": 1.50,
        "Germany": 0.80,
        "Brazil": 0.78,
        "United Kingdom": 0.65,
        "Japan": 0.90,
        "Canada": 0.75
    }
    region_factor = region_factor_map.get(region, 1.0)

    baseline_prediction = 430.0 * region_factor

    if country_model is not None:
        try:
            input_df = pd.DataFrame([{
                "year": year,
                "lag_1": lag_1 * region_factor,
                "lag_2": lag_2 * region_factor,
                "lag_3": lag_3 * region_factor,
                "rolling_mean_3": rolling_mean_3 * region_factor
            }])
            baseline_prediction = float(country_model.predict(input_df)[0])
        except Exception:
            pass

    scenario_prediction = baseline_prediction * (1 - reduction_factor)
    current_emission = baseline_prediction * 0.92

    return current_emission, baseline_prediction, scenario_prediction


def predict_sector_impacts(total_baseline, total_scenario, renewable, efficiency, ev):
    transport_share = 0.28
    energy_share = 0.34
    industry_share = 0.24
    residential_share = 0.14

    baseline = {
        "Transport": total_baseline * transport_share,
        "Energy": total_baseline * energy_share,
        "Industry": total_baseline * industry_share,
        "Residential": total_baseline * residential_share
    }

    scenario = {
        "Transport": baseline["Transport"] * (1 - ev / 180),
        "Energy": baseline["Energy"] * (1 - renewable / 150),
        "Industry": baseline["Industry"] * (1 - efficiency / 160),
        "Residential": baseline["Residential"] * (1 - renewable / 250)
    }

    # rescale to match scenario total approximately
    scale = total_scenario / max(sum(scenario.values()), 1e-6)
    scenario = {k: v * scale for k, v in scenario.items()}

    return baseline, scenario


def best_policy_mix(ev, renewable, efficiency, carbon_tax, coal_to_solar):
    weighted_scores = {
        "EV Adoption": ev * 0.10,
        "Renewable Energy": renewable * 0.28,
        "Industrial Efficiency": efficiency * 0.18,
        "Carbon Tax": carbon_tax * 0.12,
        "Coal-to-Solar": coal_to_solar * 0.22,
    }
    sorted_policies = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = [name for name, _ in sorted_policies[:3]]
    return ", ".join(top_3)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🌿 CarbonAI Controls")

region = st.sidebar.selectbox(
    "📍 Select Region",
    ["India", "United States", "China", "Germany", "Brazil", "United Kingdom", "Japan", "Canada"]
)

year_range = st.sidebar.slider(
    "📅 Year Range",
    min_value=2015,
    max_value=2035,
    value=(2020, 2030)
)

st.sidebar.markdown("### 🛠 Policy Scenario Controls")

ev_adoption = st.sidebar.slider("🚗 EV Adoption (%)", 0, 100, 35)
renewable_energy = st.sidebar.slider("☀️ Renewable Energy (%)", 0, 100, 45)
industrial_efficiency = st.sidebar.slider("🏭 Industrial Efficiency (%)", 0, 100, 30)
carbon_tax = st.sidebar.slider("💰 Carbon Tax Level (%)", 0, 100, 25)
coal_to_solar = st.sidebar.slider("🔄 Coal-to-Solar Transition (%)", 0, 100, 40)

optimize_button = st.sidebar.button("🚀 Optimize Policy", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("CarbonAI • Climate Policy Decision Engine")

# =========================================================
# COMPUTE SCENARIO
# =========================================================
if optimize_button:
    with st.spinner("Running AI policy simulation..."):
        time.sleep(1.4)

reduction_factor = calculate_policy_reduction(
    ev_adoption,
    renewable_energy,
    industrial_efficiency,
    carbon_tax,
    coal_to_solar
)

current_emissions, baseline_2030, scenario_2030 = predict_country_emission(
    region=region,
    end_year=year_range[1],
    reduction_factor=reduction_factor
)

reduction_pct = ((baseline_2030 - scenario_2030) / baseline_2030) * 100 if baseline_2030 else 0
risk_label, risk_class = get_risk_level(scenario_2030)

years_hist, hist_vals, years_forecast, baseline_vals, scenario_vals = build_forecast_series(
    current_emissions,
    reduction_factor,
    year_range[0],
    year_range[1]
)

sector_baseline, sector_scenario = predict_sector_impacts(
    baseline_2030,
    scenario_2030,
    renewable_energy,
    industrial_efficiency,
    ev_adoption
)

recommendation_mix = best_policy_mix(
    ev_adoption,
    renewable_energy,
    industrial_efficiency,
    carbon_tax,
    coal_to_solar
)

time_to_impact = "2–4 years" if renewable_energy + coal_to_solar > 90 else "4–6 years"

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero-card">
    <h1 style="margin:0; font-size:2rem;">CarbonAI Dashboard</h1>
    <p style="margin:0.35rem 0 0 0; font-size:1rem; opacity:0.95;">
        AI-powered Climate Policy Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("🌍 Current Emissions", f"{current_emissions:.1f} kt")

with m2:
    st.metric("📉 Predicted Emissions (2030)", f"{scenario_2030:.1f} kt", delta=f"-{abs(reduction_pct):.1f}%")

with m3:
    st.metric("✅ Reduction %", f"{reduction_pct:.1f}%")

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.95rem; color:#475569; margin-bottom:0.4rem;">⚠️ Risk Level</div>
        <div class="{risk_class}">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# =========================================================
# KEY INSIGHT
# =========================================================
st.markdown(f"""
<div class="insight-card">
    <strong>💡 Key Insight:</strong>
    Increasing renewable energy to <strong>{renewable_energy}%</strong> and coal-to-solar transition to
    <strong>{coal_to_solar}%</strong> reduces projected emissions in <strong>{region}</strong> by
    approximately <strong>{reduction_pct:.1f}%</strong> under this scenario.
</div>
""", unsafe_allow_html=True)

# =========================================================
# EMISSIONS FORECAST
# =========================================================
st.markdown('<div class="section-title">📈 Emissions Forecast</div>', unsafe_allow_html=True)

forecast_fig = go.Figure()

forecast_fig.add_trace(go.Scatter(
    x=years_hist,
    y=hist_vals,
    mode="lines+markers",
    name="Historical Emissions",
    line=dict(color="#0f766e", width=3)
))

forecast_fig.add_trace(go.Scatter(
    x=years_forecast,
    y=baseline_vals,
    mode="lines+markers",
    name="Baseline Forecast",
    line=dict(color="#486d9f", width=3, dash="dash")
))

forecast_fig.add_trace(go.Scatter(
    x=years_forecast,
    y=scenario_vals,
    mode="lines+markers",
    name="Policy-adjusted Forecast",
    line=dict(color="#16a34a", width=4)
))

forecast_fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=20, b=10),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#1e293b", size=14),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(color="#334155", size=12)
    ),
    xaxis=dict(
        title="Year",
        title_font=dict(color="#1e293b"),
        tickfont=dict(color="#475569")
    ),
    yaxis=dict(
        title="Emissions (kt)",
        title_font=dict(color="#1e293b"),
        tickfont=dict(color="#475569")
    )
)

st.plotly_chart(forecast_fig, use_container_width=True)

# =========================================================
# COMPARISON + SECTOR IMPACT
# =========================================================
left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.markdown('<div class="section-title">📊 Policy Impact Comparison</div>', unsafe_allow_html=True)

    compare_df = pd.DataFrame({
        "Scenario": ["Baseline 2030", "Policy Scenario 2030"],
        "Emissions": [baseline_2030, scenario_2030]
    })

    compare_fig = px.bar(
        compare_df,
        x="Scenario",
        y="Emissions",
        color="Scenario",
        text="Emissions",
        color_discrete_sequence=["#4b89df", "#16a34a"]
    )
    compare_fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    compare_fig.update_layout(
        height=380,
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1e293b", size=14),
        xaxis=dict(
            title_font=dict(color="#1e293b"),
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            title="Emissions (kt)",
            title_font=dict(color="#1e293b"),
            tickfont=dict(color="#475569")
        )
    )

    st.plotly_chart(compare_fig, use_container_width=True)

    st.info(f"Projected reduction compared to baseline: **{reduction_pct:.1f}%**")

with right_col:
    st.markdown('<div class="section-title">🏙 Sector-wise Impact</div>', unsafe_allow_html=True)

    sector_df = pd.DataFrame({
        "Sector": list(sector_scenario.keys()),
        "Scenario Emissions": list(sector_scenario.values())
    })

    sector_fig = px.pie(
        sector_df,
        names="Sector",
        values="Scenario Emissions",
        hole=0.45,
        color_discrete_sequence=["#16a34a", "#0f766e", "#65a30d", "#86efac"]
    )
    sector_fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="white",
        font=dict(color="#1e293b", size=14),
        legend=dict(font=dict(color="#334155", size=12))
    )
    st.plotly_chart(sector_fig, use_container_width=True)

# =========================================================
# AI RECOMMENDATION + RISK
# =========================================================
col_a, col_b = st.columns([1.2, 0.8])

with col_a:
    st.markdown('<div class="section-title">🤖 AI Recommendation Panel</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="recommend-card">
        <h4 style="margin-top:0;">Best policy mix</h4>
        <p style="margin-bottom:0.5rem;">
            <strong>{recommendation_mix}</strong>
        </p>
        <p style="margin:0.25rem 0;"><strong>Expected emission reduction:</strong> {reduction_pct:.1f}%</p>
        <p style="margin:0.25rem 0;"><strong>Time to impact:</strong> {time_to_impact}</p>
        <p style="margin:0.5rem 0 0 0;">
            Recommendation prioritizes energy transition and structural efficiency for maximum regional effect.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-title">🚨 Region Risk Indicator</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <p style="margin:0 0 0.6rem 0; color:#475569;">Region: <strong>{region}</strong></p>
        <div class="{risk_class}">{risk_label} Risk</div>
        <p style="margin-top:0.9rem; color:#475569;">
            Risk is estimated from projected 2030 emissions under the selected policy scenario.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# OPTIONAL MAP / PLACEHOLDER
# =========================================================
with st.expander("🗺 Region Emissions View"):
    map_df = pd.DataFrame({
        "Region": ["India", "United States", "China", "Germany", "Brazil", "United Kingdom", "Japan", "Canada"],
        "Emissions": [520, 490, 610, 240, 260, 190, 310, 220]
    })

    map_fig = px.bar(
        map_df,
        x="Region",
        y="Emissions",
        color="Emissions",
        color_continuous_scale="Greens"
    )
    map_fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1e293b", size=14),
        xaxis=dict(
            title_font=dict(color="#1e293b"),
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            title_font=dict(color="#1e293b"),
            tickfont=dict(color="#475569")
        )
    )
    st.plotly_chart(map_fig, use_container_width=True)
    st.caption("Map placeholder can be replaced with a real geo-visual later.")

# =========================================================
# MODEL STATUS
# =========================================================
with st.expander("⚙️ Model Connection Status"):
    st.write("This dashboard is designed to connect with your saved models.")
    st.write(f"Country model loaded: {'Yes' if country_model is not None else 'No'}")
    st.write(f"Industry model loaded: {'Yes' if industry_model is not None else 'No'}")
    st.write("If a model is unavailable, the dashboard uses realistic simulation logic so the UI remains demo-ready.")