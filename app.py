import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="CarbonAI — Industrial Carbon Intelligence Platform",
    page_icon="🌍",
    layout="wide",
)


# --------- Theme and layout ---------
DARK_CSS = """
<style>
body {
    background-color: #06152a;
    color: #e8f6ff;
}
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #052038 0%, #0b2e4a 100%);
    border-right: 1px solid rgba(96, 165, 250, 0.18);
}
.stApp {
    background: linear-gradient(180deg, #06152a 0%, #0d3551 100%);
}
section.main {
    background-color: transparent;
}
div[data-testid='metric-container'] {
    background: rgba(8, 24, 42, 0.94) !important;
    border: 1px solid rgba(56, 189, 248, 0.18) !important;
    border-radius: 20px !important;
    padding: 18px !important;
    box-shadow: 0 18px 40px rgba(8, 24, 42, 0.35) !important;
}
.stButton>button {
    background-color: #22c55e !important;
    color: #020617 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.4rem !important;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #0f766e !important;
}
.stTextInput>div>div>input,
.stNumberInput>div>div>input,
.stSelectbox>div>div>div>div,
.stSlider>div>div>input {
    background-color: #0a1c32 !important;
    color: #e8f6ff !important;
    border: 1px solid rgba(56, 189, 248, 0.18) !important;
}
.css-1d391kg,
.css-1lsmho7,
.css-1offfwp {
    background-color: rgba(10, 23, 40, 0.95) !important;
}
.stApp .css-1d391kg, .stApp .css-1lsmho7 {
    border-radius: 20px;
}
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc;
}
span[data-testid='stMarkdownContainer'] p {
    color: #cbd5e1;
}
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)

# --------- Data loading ---------
@st.cache_data(show_spinner=False)
def load_emission_data() -> pd.DataFrame:
    path = DATA_DIR / "carbon_emission_dataset_with_Industry.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Renewable_Share"] = np.round((df["Renewable_Energy_Consumption_kWh"] / df["Total_Energy_Consumption_kWh"]) * 100, 1)
    return df


@st.cache_data(show_spinner=False)
def load_country_data() -> pd.DataFrame:
    path = DATA_DIR / "co2_emissions_kt_by_country.csv"
    df = pd.read_csv(path)
    return df.dropna(subset=["value"])


@st.cache_resource
def load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception as error:
        return None


emission_df = load_emission_data()
country_df = load_country_data()
carbon_model = load_model(MODELS_DIR / "carbon_model.joblib")
forecast_model = load_model(MODELS_DIR / "country_forecast.joblib")

# --------- Helper functions ---------

def risk_level(prediction: float) -> tuple[str, str]:
    if prediction < 20:
        return "Low Risk", "🟢"
    if prediction < 40:
        return "Medium Risk", "🟡"
    return "High Risk", "🔴"


def sustainability_score(efficiency: float, renewable_share: float, nonrenewable: float) -> int:
    score = efficiency * 0.4 + renewable_share * 0.4 - (nonrenewable / 100) * 0.2
    return int(np.clip(score, 0, 100))


def generate_recommendations(renewable: float, efficiency: float, transport_km: float, transport_mode: str) -> list[str]:
    suggestions = []
    if renewable < 0.35:
        suggestions.append("Increase renewable energy generation by shifting to solar or wind.")
    if efficiency < 75:
        suggestions.append("Enhance process efficiency across the plant.")
    if transport_km > 1200:
        suggestions.append("Shorten transport routes and optimize logistics.")
    if transport_mode in ["Air", "Ship"]:
        suggestions.append("Switch to lower-carbon logistics such as rail or electric vehicles.")
    return suggestions


def plotly_dark_layout(fig: go.Figure):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#081127",
        font_color="#e2e8f0",
        margin=dict(l=0, r=0, t=40, b=20),
        legend=dict(font=dict(color="#cbd5e1")),
        title_font=dict(color="#f8fafc", size=18),
    )
    return fig


def display_metrics():
    total_records = len(emission_df)
    unique_companies = emission_df["Company_ID"].nunique()
    sectors = emission_df["Sector"].nunique()
    avg_emission = emission_df["Carbon_Emission_tCO2e_TARGET"].mean()
    renewable_share = emission_df["Renewable_Share"].mean()

    col1, col2, col3, col4 = st.columns(4, gap="large")
    col1.metric("Records", f"{total_records:,}")
    col2.metric("Companies", f"{unique_companies}")
    col3.metric("Sectors", f"{sectors}")
    col4.metric("Avg Emission", f"{avg_emission:.2f} tCO₂e")

    st.markdown("---")
    c1, c2, c3 = st.columns(3, gap="large")
    c1.metric("Renewable Share", f"{renewable_share:.1f}%")
    c2.metric("Mean Efficiency", f"{emission_df['Process_Efficiency_Percent'].mean():.1f}%")
    c3.metric("Forecast Model", "Loaded" if forecast_model is not None else "Missing")


# --------- Page renderers ---------

def render_dashboard():
    st.markdown("# CarbonAI — Industrial Carbon Intelligence Platform")
    st.markdown("Operational carbon intelligence for industrial decarbonization, governance, and strategic planning.")

    st.markdown("---")
    hero_left, hero_right = st.columns([3, 1], gap="large")
    with hero_left:
        st.markdown("### Executive summary")
        st.markdown(
            "Carbon AI transforms industrial operations into measurable carbon performance. Use this platform to evaluate emissions by sector, model future scenarios, and identify decarbonization opportunities with confidence."
        )
        st.markdown(
            "<div style='padding:18px; border-radius:20px; background: rgba(14, 28, 50, 0.85); border:1px solid rgba(148,163,184,0.12)'>"
            "<p style='margin:0 0 8px; color:#7dd3fc; font-weight:700'>Current operating profile</p>"
            "<p style='margin:2px 0 0; color:#e2e8f0;'>Integrated emissions model + country forecasting + recommendations</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with hero_right:
        st.metric(label="Model health", value="Stable")
        st.metric(label="Forecast horizon", value="2035")
        st.metric(label="UI theme", value="Enterprise dark")

    st.markdown("---")
    display_metrics()

    st.markdown("### Operational analytics")
    hist_col, bar_col = st.columns(2, gap="large")
    with hist_col:
        fig = px.violin(
            emission_df,
            y="Renewable_Share",
            title="Renewable energy share distribution",
            color_discrete_sequence=["#22c55e"],
            box=True,
            points="all",
        )
        fig.update_yaxes(title_text="Renewable Share (%)")
        plotly_dark_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with bar_col:
        sector_summary = (
            emission_df.groupby("Sector")["Carbon_Emission_tCO2e_TARGET"].mean().sort_values(ascending=False).reset_index()
        )
        fig2 = px.bar(
            sector_summary.head(10),
            x="Carbon_Emission_tCO2e_TARGET",
            y="Sector",
            orientation="h",
            title="Average emission by sector",
            color="Carbon_Emission_tCO2e_TARGET",
            color_continuous_scale=["#22c55e", "#38bdf8"],
        )
        plotly_dark_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Key outcomes")
    outcome_1, outcome_2, outcome_3 = st.columns(3, gap="large")
    outcome_1.metric("Emission visibility", "Real-time insights")
    outcome_2.metric("Risk scoring", "Operational readiness")
    outcome_3.metric("Strategic planning", "Carbon reduction")



def render_prediction():
    st.markdown("# Predict Industrial Carbon Emissions")
    st.markdown("Enter industry parameters below, then run the model to generate a carbon emission estimate and maturity score.")

    with st.expander("Input guidance", expanded=True):
        st.write(
            "Use actual operational metrics where possible. The model is trained on industrial energy, transport, and process efficiency features to provide realistic carbon estimates."
        )
        st.markdown(
            "- Choose a representative sector and transport mode.\n"
            "- Keep renewable energy and efficiency values aligned with current operations.\n"
            "- The prediction is designed for industrial sustainability planning and executive reporting."
        )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Industry inputs")
        sector = st.selectbox("Sector", sorted(emission_df["Sector"].unique()))
        industry_sector = st.selectbox("Industry Sector", sorted(emission_df["Industry_Sectors"].unique()))
        energy = st.number_input("Total Energy Consumption (kWh)", min_value=0.0, value=100000.0, step=500.0)
        renewable = st.number_input("Renewable Energy (kWh)", min_value=0.0, value=40000.0, step=500.0)
        nonrenewable = st.number_input("Non-Renewable Energy (kWh)", min_value=0.0, value=60000.0, step=500.0)
    with right:
        st.subheader("Operational inputs")
        production = st.number_input("Production Output Units", min_value=0.0, value=5000.0, step=100.0)
        transport_mode = st.selectbox("Transport Mode", sorted(emission_df["Supply_Chain_Transport_Mode"].unique()))
        transport_km = st.number_input("Transport Distance (km)", min_value=0.0, value=1200.0, step=50.0)
        efficiency = st.slider("Process Efficiency (%)", 0, 100, 72)
        renewable_share = st.slider("Expected Renewable Share (%)", 0, 100, 36)
        strategy = st.selectbox(
            "Carbon Reduction Strategy",
            sorted(emission_df["Carbon_Reduction_Strategy"].unique()),
        )

    st.markdown("---")
    if st.button("Run prediction"):
        if carbon_model is None:
            st.error("Prediction model is unavailable. Please check the model files.")
            return

        input_df = pd.DataFrame(
            {
                "Sector": [sector],
                "Total_Energy_Consumption_kWh": [energy],
                "Renewable_Energy_Consumption_kWh": [renewable],
                "NonRenewable_Energy_Consumption_kWh": [nonrenewable],
                "Production_Output_Units": [production],
                "Supply_Chain_Transport_km": [transport_km],
                "Supply_Chain_Transport_Mode": [transport_mode],
                "Raw_Material_Usage_kg": [nonrenewable * 0.9],
                "Energy_Cost_USD": [energy * 0.05],
                "Carbon_Tax_USD": [100],
                "Process_Efficiency_Percent": [efficiency],
                "Employment_Count": [1500],
                "Public_Acceptance_Index": [75],
                "Carbon_Reduction_Strategy": [strategy],
                "Strategy_Implementation_Cost_USD": [200000],
                "Expected_Carbon_Reduction_Percent": [renewable_share * 0.25],
                "Expected_Renewable_Share_Percent": [renewable_share],
                "Social_Impact_Score": [80],
                "Industry_Sectors": [industry_sector],
            }
        )

        predicted = carbon_model.predict(input_df)[0]
        label, symbol = risk_level(predicted)
        score = sustainability_score(efficiency, renewable_share, nonrenewable)
        recommendations = generate_recommendations(renewable / energy if energy else 0, efficiency, transport_km, transport_mode)

        st.markdown("### Prediction results")
        cards = st.columns(3, gap="large")
        cards[0].metric("Predicted Emission", f"{predicted:.2f} tCO₂e")
        cards[1].metric("Carbon Risk", f"{symbol} {label}")
        cards[2].metric("Sustainability Score", f"{score}/100")

        st.markdown("### Recommended actions")
        with st.expander("View recommended actions", expanded=True):
            if recommendations:
                for rec in recommendations:
                    st.success(rec)
            else:
                st.info("This scenario is aligned with a low-risk carbon profile.")

        gauge, chart = st.columns([1, 1], gap="large")
        with gauge:
            fig = go.Figure(
                data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    delta={'reference': 65, 'increasing': {'color': '#22c55e'}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': '#22c55e'},
                        'bgcolor': '#0b1120',
                    },
                    title={'text': "Sustainability Index"},
                )]
            )
            plotly_dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

        with chart:
            trend_df = emission_df.groupby("Sector")["Carbon_Emission_tCO2e_TARGET"].mean().reset_index().sort_values(by="Carbon_Emission_tCO2e_TARGET", ascending=False)
            fig2 = px.bar(
                trend_df.head(8),
                x="Carbon_Emission_tCO2e_TARGET",
                y="Sector",
                orientation="h",
                title="Sector emission benchmark",
                color="Carbon_Emission_tCO2e_TARGET",
                color_continuous_scale=["#22c55e", "#38bdf8"],
            )
            plotly_dark_layout(fig2)
            st.plotly_chart(fig2, use_container_width=True)


def render_analytics():
    st.markdown("# Global Carbon Analytics")
    st.markdown("A premium analytics suite tracking global emissions and country-level performance.")

    latest_year = int(country_df["year"].max())
    latest_df = country_df[country_df["year"] == latest_year]
    top10 = latest_df.nlargest(10, "value")

    kpi1, kpi2, kpi3 = st.columns(3, gap="large")
    kpi1.metric("Latest year", latest_year)
    kpi2.metric("Top emitter", top10.iloc[0]["country_name"])
    kpi3.metric("Global CO₂", f"{country_df['value'].sum():,.0f} kt")

    st.markdown("---")
    st.markdown("### Top emitters")
    fig = px.bar(
        top10,
        x="country_name",
        y="value",
        color="value",
        color_continuous_scale=["#22c55e", "#0ea5e9"],
        title=f"Top 10 CO₂ Emitters ({latest_year})",
    )
    plotly_dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        country = st.selectbox("Country trend", ["India", "China", "United States", "Russia", "Japan"])
        country_trend = country_df[country_df["country_name"] == country]
        fig2 = px.line(country_trend, x="year", y="value", markers=True, color_discrete_sequence=["#38bdf8"], title=f"{country} CO₂ Emission Trend")
        plotly_dark_layout(fig2)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        comparison = country_df[country_df["country_name"].isin(["India", "China"])]
        fig3 = px.line(comparison, x="year", y="value", color="country_name", markers=True, color_discrete_sequence=["#22c55e", "#38bdf8"], title="India vs China")
        plotly_dark_layout(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("### Global trend")
    total_by_year = country_df.groupby("year")["value"].sum().reset_index()
    fig4 = px.area(total_by_year, x="year", y="value", title="Global CO₂ Emissions Over Time", color_discrete_sequence=["#22c55e"])
    plotly_dark_layout(fig4)
    st.plotly_chart(fig4, use_container_width=True)


def render_forecast():
    st.markdown("# Forecast & Strategy")
    st.markdown("Generate future CO₂ emission scenarios and validate long-term decarbonization progress.")

    if forecast_model is None:
        st.error("Forecast model is unavailable. Please check the saved forecast model.")
        return

    year = st.slider("Forecast Year", 2026, 2035, 2030)
    prediction = forecast_model.predict(pd.DataFrame({"year": [year]}))[0]

    st.markdown("---")
    st.markdown("### Forecast output")
    st.metric("Forecasted CO₂ (kt)", f"{prediction:,.2f}")

    projected_years = np.arange(2026, 2036)
    future = pd.DataFrame({"year": projected_years})
    future["forecast"] = forecast_model.predict(future)

    fig = px.line(future, x="year", y="forecast", markers=True, title="Forecasted CO₂ Trend")
    plotly_dark_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Scenario summary")
    st.write(
        "This forecast uses the saved model to project emissions across the next decade. Use the results to align carbon targets with operational plans."
    )

    st.divider()
    st.markdown("### Forecast details")
    st.dataframe(future.style.format({"forecast": "{:.2f}"}))


def render_about():
    st.markdown("# About CarbonAI")
    st.markdown(
        "CarbonAI delivers enterprise-grade industrial carbon intelligence—combining predictive modeling, sector benchmarking, and actionable decarbonization guidance.")

    st.markdown("### Problem Statement")
    st.write(
        "Industrial emissions are a leading contributor to climate risk, and organizations need a predictive carbon intelligence platform "
        "for operations, strategy, and sustainability reporting."
    )

    st.markdown("### Business Value")
    st.write(
        "This platform converts operational metrics into actionable carbon forecasts, risk scores, and sustainability recommendations. "
        "It is designed to support green investment decisions and corporate decarbonization programs."
    )

    st.markdown("### Architecture")
    st.write(
        "- CSV datasets provide real-world carbon emissions and industrial energy profiles.\n"
        "- Scikit-Learn models deliver prediction and forecast outputs.\n"
        "- Plotly charts enable interactive executive dashboards.\n"
        "- Streamlit provides a modern SaaS UI with responsive layout and dark theme."
    )

    st.markdown("### Tech Stack")
    st.write("Python, Streamlit, Pandas, NumPy, Plotly, Scikit-Learn, Joblib")

    st.markdown("### Future Scope")
    st.write(
        "- Add multi-factor scenario planning and KPI targets.\n"
        "- Extend global analytics with emissions intensity and sector benchmarking.\n"
        "- Add user authentication, reports, and deployment to a cloud SaaS environment."
    )


def main():
    st.sidebar.title("CarbonAI")
    st.sidebar.markdown("Industrial Carbon Intelligence & Sustainability Analytics Suite")
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Prediction", "Analytics", "Forecast", "About"],
        index=0,
    )

    if page == "Dashboard":
        render_dashboard()
    elif page == "Prediction":
        render_prediction()
    elif page == "Analytics":
        render_analytics()
    elif page == "Forecast":
        render_forecast()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
