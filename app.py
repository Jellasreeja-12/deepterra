# ============================================================
#   DeepTerra — app.py (Streamlit Cloud Compatible)
#   No TensorFlow — uses scikit-learn ANN
# ============================================================

import warnings
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DeepTerra — Landslide Early Warning",
    page_icon="🏔️", layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    .main-title  { font-size:2.8rem; font-weight:900; color:#ffffff; letter-spacing:-1px; }
    .sub-title   { color:#888; font-size:1rem; margin-bottom:1rem; }
    .safe-box    { background:#0d2b0d; border:1px solid #2d7a2d; border-left:6px solid #4CAF50; padding:1.2rem; border-radius:12px; }
    .risk-box    { background:#2b0d0d; border:1px solid #7a2d2d; border-left:6px solid #F44336; padding:1.2rem; border-radius:12px; }
    .warn-box    { background:#2b240d; border:1px solid #7a6a2d; border-left:6px solid #FFC107; padding:1.2rem; border-radius:12px; }
    .metric-card { background:#1a1a2e; border:1px solid #2a2a4e; padding:1rem; border-radius:12px; text-align:center; }
    .section-header { font-size:1.1rem; font-weight:700; color:#a0a0ff; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = pickle.load(open("model/deepterra_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl",          "rb"))
    metadata = pickle.load(open("model/metadata.pkl",        "rb"))
    return model, scaler, metadata


try:
    model, scaler, metadata = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Model not found. Run train.py first.\nError: {e}")
    st.stop()

feature_cols = metadata['feature_cols']
shap_dict = metadata['shap_dict']

# ── HEADER ──
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">🏔️ DeepTerra</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explainable AI-Based Landslide Risk Prediction & Early Warning System</div>', unsafe_allow_html=True)
with col_h2:
    st.markdown('<br>', unsafe_allow_html=True)
    st.success(f"Model Accuracy: **{metadata['accuracy']}%**")
    st.info(f"F1 Score: **{metadata['f1']}%**")

st.divider()

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-header">📍 Environmental Parameters</div>',
                unsafe_allow_html=True)
    temperature = st.slider("🌡️ Temperature (°C)",   0,   50,  25)
    humidity = st.slider("💧 Humidity (%)",         0,  100,  70)
    precipitation = st.slider("🌧️ Precipitation (mm)",
                              0,  400, 120, help="Expected rainfall amount")
    soil_moisture = st.slider("🌱 Soil Moisture (%)",   0,  100,  60)
    elevation = st.slider("⛰️ Elevation (m)",        0, 3000, 500)

    st.divider()
    with st.expander("📌 High Risk Reference Zones in India"):
        st.markdown("""
        | Location | Typical Conditions |
        |---|---|
        | Wayanad, Kerala | Precip >200mm, Soil >80%, Elev 700-900m |
        | Uttarakhand | Precip >180mm, Humidity >85%, Elev >1500m |
        | Himachal Pradesh | Precip >150mm, Soil >75%, Elev >1200m |
        | Manipur | Precip >160mm, Humidity >80%, Elev >600m |
        """)
    predict_btn = st.button("🔍 Predict Landslide Risk",
                            use_container_width=True, type="primary")

with right_col:
    if predict_btn:
        input_data = np.array(
            [[temperature, humidity, precipitation, soil_moisture, elevation]])
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]
        prediction = 1 if prob >= 0.5 else 0

        if prob < 0.4:
            risk_level = "🟢 LOW"
            css_class = "safe-box"
            advice = "Conditions appear safe. Continue routine monitoring."
        elif prob < 0.65:
            risk_level = "🟡 MODERATE"
            css_class = "warn-box"
            advice = "Conditions are concerning. Authorities should stay alert and monitor closely."
        else:
            risk_level = "🔴 HIGH"
            css_class = "risk-box"
            advice = "HIGH LANDSLIDE RISK. Immediate action recommended. Consider evacuation warnings."

        st.markdown(
            '<div class="section-header">🚨 Prediction Result</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.8rem;font-weight:900;">{prob*100:.1f}%</div><div style="color:#888;font-size:0.8rem;">Probability</div></div>', unsafe_allow_html=True)
        with c2:
            verdict = "⚠️ AT RISK" if prediction == 1 else "✅ SAFE"
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.4rem;font-weight:900;">{verdict}</div><div style="color:#888;font-size:0.8rem;">Verdict</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="metric-card"><div style="font-size:1.2rem;font-weight:900;">{risk_level}</div><div style="color:#888;font-size:0.8rem;">Risk Level</div></div>', unsafe_allow_html=True)

        st.markdown(
            f'<br><div class="{css_class}"><b>{risk_level} RISK</b><br>{advice}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(prob), text=f"Risk Probability: {prob*100:.1f}%")

        # ── SHAP EXPLAINABILITY ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="section-header">🧠 Explainable AI — Why This Prediction?</div>', unsafe_allow_html=True)

        input_dict = {
            'Temperature (°C)':   temperature,
            'Humidity (%)':       humidity,
            'Precipitation (mm)': precipitation,
            'Soil Moisture (%)':  soil_moisture,
            'Elevation (m)':      elevation
        }
        importance_df = pd.DataFrame({
            'Feature':    list(shap_dict.keys()),
            'Importance': list(shap_dict.values()),
            'Your Input': [input_dict[f] for f in shap_dict.keys()]
        }).sort_values('Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        imp_vals = importance_df['Importance'].apply(
            lambda x: float(x[0]) if isinstance(x, list) else float(x))
        colors = ['#F44336' if v > imp_vals.mean(
        ) else '#4CAF50' for v in imp_vals]
        importance_df['Importance'] = imp_vals
        ax.barh(importance_df['Feature'],
                importance_df['Importance'], color=colors)
        ax.set_xlabel('Mean |SHAP Value|', color='white')
        ax.set_title('Feature Contribution to Prediction',
                     color='white', fontsize=11)
        ax.tick_params(colors='white')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#444')
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        top_feature = importance_df.iloc[0]['Feature']
        top_value = importance_df.iloc[0]['Your Input']
        st.markdown(f"""
        <div style='background:#1a1a2e;padding:1rem;border-radius:10px;border:1px solid #2a2a4e;margin-top:0.5rem;'>
        <b>🔍 Key Factor:</b> <span style='color:#a0a0ff'>{top_feature}</span> is the most influential feature.<br>
        <b>Your value:</b> {top_value} &nbsp;|&nbsp; <b>Impact score:</b> {importance_df.iloc[0]['Importance']:.4f}
        </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Input Summary"):
            st.dataframe(importance_df.set_index(
                'Feature'), use_container_width=True)
    else:
        st.markdown(
            '<div class="section-header">🚨 Prediction Result</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#1a1a2e;border:1px dashed #333;border-radius:12px;
                    padding:3rem;text-align:center;color:#555;'>
            <div style='font-size:3rem;'>🏔️</div>
            <div style='margin-top:1rem;'>Adjust parameters on the left<br>and click <b>Predict</b> to see results</div>
        </div>""", unsafe_allow_html=True)

st.divider()
st.markdown("<center><small>DeepTerra · Explainable AI · ANN + SHAP · Landslide Early Warning · 2026</small></center>", unsafe_allow_html=True)
