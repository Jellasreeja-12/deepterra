# ============================================================
#   DeepTerra — app.py
#   Explainable AI-Based Landslide Risk Prediction
#   Streamlit Dashboard
# ============================================================

import matplotlib
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="DeepTerra — Landslide Early Warning",
    page_icon="🏔️",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    .main-title  { font-size:2.8rem; font-weight:900; color:#ffffff; letter-spacing:-1px; }
    .sub-title   { color:#888; font-size:1rem; margin-bottom:1rem; }
    .safe-box    { background:#0d2b0d; border:1px solid #2d7a2d; border-left:6px solid #4CAF50;
                   padding:1.2rem; border-radius:12px; }
    .risk-box    { background:#2b0d0d; border:1px solid #7a2d2d; border-left:6px solid #F44336;
                   padding:1.2rem; border-radius:12px; }
    .warn-box    { background:#2b240d; border:1px solid #7a6a2d; border-left:6px solid #FFC107;
                   padding:1.2rem; border-radius:12px; }
    .metric-card { background:#1a1a2e; border:1px solid #2a2a4e; padding:1rem;
                   border-radius:12px; text-align:center; }
    .section-header { font-size:1.1rem; font-weight:700; color:#a0a0ff;
                      letter-spacing:0.05em; text-transform:uppercase;
                      margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ───────────────────────────────────────────


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model/deepterra_ann.keras")
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    metadata = pickle.load(open("model/metadata.pkl", "rb"))
    return model, scaler, metadata


try:
    model, scaler, metadata = load_assets()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# ── FIX FEATURE ORDER SAFETY ─────────────────────────────
feature_cols = list(metadata["feature_cols"])
shap_dict = dict(zip(feature_cols, metadata["shap_dict"].values()))

# ── HEADER ───────────────────────────────────────────────
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="main-title">🏔️ DeepTerra</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Landslide Risk Prediction System</div>',
                unsafe_allow_html=True)

with col2:
    st.success(f"Accuracy: {metadata['accuracy']}%")
    st.info(f"F1 Score: {metadata['f1']}%")

st.divider()

# ── INPUTS ───────────────────────────────────────────────
left, right = st.columns(2)

with left:
    temperature = st.slider("Temperature (°C)", 0, 50, 22)
    humidity = st.slider("Humidity (%)", 0, 100, 70)
    precipitation = st.slider("Rainfall (mm)", 0, 400, 120)
    soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 60)
    elevation = st.slider("Elevation (m)", 0, 3000, 500)

    predict_btn = st.button("Predict", type="primary")

with right:
    st.info("Adjust values and click Predict")

# ── PREDICTION ───────────────────────────────────────────
if predict_btn:

    input_data = np.array(
        [[temperature, humidity, precipitation, soil_moisture, elevation]])

    try:
        input_scaled = scaler.transform(input_data)
        prob = float(model.predict(input_scaled, verbose=0)[0][0])
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    prediction = 1 if prob >= 0.5 else 0

    if prob < 0.4:
        risk = "🟢 LOW"
        msg = "Safe conditions"
    elif prob < 0.65:
        risk = "🟡 MODERATE"
        msg = "Stay alert"
    else:
        risk = "🔴 HIGH"
        msg = "Danger zone"

    # ── OUTPUT ───────────────────────────────────────────
    st.subheader("Result")

    st.metric("Probability", f"{prob*100:.2f}%")
    st.metric("Risk Level", risk)
    st.write(msg)

    st.progress(min(max(prob, 0.0), 1.0))

    # ── FEATURE IMPORTANCE ──────────────────────────────
    st.subheader("Feature Impact")

    input_dict = {
        "Temperature (°C)": temperature,
        "Humidity (%)": humidity,
        "Precipitation (mm)": precipitation,
        "Soil Moisture (%)": soil_moisture,
        "Elevation (m)": elevation
    }

    df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": list(shap_dict.values()),
        "Input": [input_dict[f] for f in feature_cols]
    }).sort_values("Importance", ascending=False)

    st.dataframe(df)

# ── MODEL INSIGHTS ───────────────────────────────────────
st.divider()
st.subheader("Model Insights")

col1, col2 = st.columns(2)

with col1:
    try:
        st.image("model/shap_summary.png")
    except:
        st.warning("SHAP image missing")

with col2:
    try:
        st.image("model/confusion_matrix.png")
    except:
        st.warning("Confusion matrix missing")
