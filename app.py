# app.py
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from src.features import create_features
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Uber Trip Analysis Dashboard",
    page_icon="🚕",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
/* === BACKGROUND === */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #FFFFFF;
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background-color: rgba(15, 15, 30, 0.85);
    backdrop-filter: blur(12px);
    color: #E0E0E0;
}

/* === HEADINGS === */
h1, h2, h3, h4, h5 {
    color: #00E5FF;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
}

/* === BODY TEXT === */
p, li, label, span {
    color: #EDEDED !important;
    font-family: 'Poppins', sans-serif;
}

/* === METRIC CARDS === */
div[data-testid="stMetricValue"] {
    color: #FF4B4B;
    font-weight: bold;
    font-size: 28px;
}
div[data-testid="stMetricLabel"] {
    color: #00E5FF;
    font-weight: 500;
}

/* === CARD CONTAINERS === */
.card {
    background-color: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    backdrop-filter: blur(10px);
}

/* === BUTTONS === */
div.stButton > button {
    background: linear-gradient(90deg, #00E5FF, #FF4B4B);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 10px 25px;
    font-weight: bold;
    font-size: 16px;
    box-shadow: 0px 4px 10px rgba(0,229,255,0.4);
    transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #FF4B4B, #00E5FF);
}

/* === SEPARATORS === */
hr {
    border: 1px solid rgba(255,255,255,0.15);
    margin: 1.5rem 0;
}

/* === TABS === */
[data-testid="stTabs"] button {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
    color: #FFFFFF !important;
    font-weight: 600;
}
[data-testid="stTabs"] button:hover {
    background: rgba(255,255,255,0.25);
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(90deg, #00E5FF, #FF4B4B);
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🚕 Uber Trip Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#B0E0E6;'>Elegant • Insightful • Data-Driven</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- LOAD MODEL + METRICS ----------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.joblib")

@st.cache_data
def load_metrics():
    path = Path("reports/metrics.json")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

model = load_model()
metrics = load_metrics()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Model Insights", "🔮 Prediction"])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    st.markdown("<h3>📊 Model Performance Summary</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MAE", f"{metrics['MAE']:.2f}" if metrics else "N/A")
    with col2:
        st.metric("RMSE", f"{metrics['RMSE']:.2f}" if metrics else "N/A")
    with col3:
        st.metric("MAPE (%)", f"{metrics['MAPE']:.2f}" if metrics else "N/A")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Load trip data
    data_path = Path("data/processed/uber_hourly.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["datetime"], index_col="datetime")
        df = df.reset_index()
        fig = px.area(
            df, x="datetime", y="Count",
            title="Uber Trip Volume Over Time",
            color_discrete_sequence=["#00E5FF"],
            template="plotly_dark"
        )
        fig.update_traces(opacity=0.75)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data not found. Please add uber_hourly.csv in data/processed/")

# ---------------- TAB 2: MODEL INSIGHTS ----------------
with tab2:
    st.markdown("<h3>📈 Feature Importance & Residual Analysis</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if Path("reports/feature_importance.png").exists():
        col1.image("reports/feature_importance.png", caption="Feature Importance")
    if Path("reports/residuals.png").exists():
        col2.image("reports/residuals.png", caption="Residual Distribution")

    st.markdown("""
    <div class='card'>
        <h4 style='color:#00E5FF;'>💡 Key Insights</h4>
        <ul>
            <li>Lag_1 and Lag_7 are dominant predictors (strong weekly pattern).</li>
            <li>Low residual spread → model fits well to observed demand.</li>
            <li>Weekends consistently show ~30% higher trip volume → surge pricing opportunity.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- TAB 3: PREDICTION ----------------
with tab3:
    st.markdown("<h3>🔮 Predict Future Trip Demand</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload new Uber trip data (CSV with datetime & Count)", type=["csv"])
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file, parse_dates=["datetime"], index_col="datetime")
        df_feat = create_features(df_new)
        df_feat_latest = df_feat.tail(1)
        y_pred = model.predict(df_feat_latest)[0]
        st.success(f"Predicted Trips for Next Day: **{int(y_pred):,}**")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.write("📅 Recent Trend & Predicted Value")
        df_tail = df_new.tail(7).reset_index()
        df_tail["Type"] = "Actual"
        next_day = pd.DataFrame({
            "datetime": [df_new.index[-1] + pd.Timedelta(days=1)],
            "Count": [y_pred],
            "Type": ["Predicted"]
        })
        df_plot = pd.concat([df_tail, next_day])
        fig2 = px.line(
            df_plot, x="datetime", y="Count", color="Type",
            color_discrete_map={"Actual": "#00E5FF", "Predicted": "#FF4B4B"},
            template="plotly_dark", markers=True
        )
        fig2.update_layout(font=dict(color="#FFFFFF"))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Upload a new CSV to generate a prediction.")

# ---------------- FOOTER ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#CCCCCC;'>"
    "Developed by <b style='color:#00E5FF;'>Nanda S.C</b> | Uber Trip Analytics Project 🚀"
    "</p>",
    unsafe_allow_html=True
)
