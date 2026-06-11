# 🏔️ DeepTerra — Explainable AI-Based Landslide Risk Prediction & Early Warning System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepterra-isz9tmmyqspddym4rww4fu.streamlit.app/)

> **Live App:** https://deepterra-isz9tmmyqspddym4rww4fu.streamlit.app/

---

## 📌 About the Project

**DeepTerra** is an AI-powered landslide early warning system built for disaster management authorities in India. It predicts whether current environmental conditions indicate a landslide risk and explains *why* the prediction was made using Explainable AI (SHAP).

Incidents like the **Wayanad landslide (2024)**, **Kedarnath disaster (2013)**, and **Malin village tragedy (2014)** had one thing in common — zero early warning. DeepTerra addresses this critical gap by giving authorities actionable intelligence hours before a disaster strikes.

---

## 🎯 Objective

- Predict landslide risk (Safe / At Risk) based on real-time environmental conditions
- Explain which factors are driving the prediction using SHAP values
- Provide a clean, deployable dashboard for disaster management authorities

---

## 🧠 How It Works

```
User inputs environmental conditions
            ↓
ANN model predicts landslide probability
            ↓
SHAP explains which features caused the prediction
            ↓
Dashboard shows risk level + evacuation advice
            ↓
Authorities issue warnings and evacuate people
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 98.12% |
| Precision | 97.53% |
| Recall | 98.75% |
| F1 Score | 98.14% |

---

## 🔬 Tech Stack

| Component | Technology |
|-----------|-----------|
| Machine Learning | Scikit-learn MLPClassifier (ANN) |
| Explainable AI | SHAP (SHapley Additive Explanations) |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
DeepTerra/
├── data/
│   └── regenerated_landslide_risk_dataset.csv
├── model/
│   ├── deepterra_model.pkl
│   ├── scaler.pkl
│   ├── metadata.pkl
│   └── shap_summary.png
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Jellasreeja-12/deepterra.git
cd deepterra
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python train.py
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## 🌍 Dataset

The dataset contains environmental and geological features:

| Feature | Description |
|---------|-------------|
| Temperature (°C) | Ambient temperature |
| Humidity (%) | Relative humidity |
| Precipitation (mm) | Rainfall amount |
| Soil Moisture (%) | Soil water content |
| Elevation (m) | Terrain elevation |
| Landslide Risk | Target label (Low / Moderate / High / Very High) |

Labels are converted to binary: **Low → 0 (Safe)**, **Moderate/High/Very High → 1 (Risk)**

---

## 🧪 Explainable AI — SHAP

Unlike traditional black-box models, DeepTerra uses **SHAP (SHapley Additive Explanations)** to explain every prediction. Disaster authorities can see exactly which environmental factors triggered the warning — making the system transparent and trustworthy.

Key contributing factors identified:
- **Humidity (%)** — most influential feature
- **Soil Moisture (%)** — second most influential
- **Precipitation (mm)** — third most influential

---

## 🏔️ High Risk Zones in India

| Location | Risk Level |
|----------|-----------|
| Wayanad, Kerala | 🔴 Very High |
| Uttarakhand | 🔴 Very High |
| Himachal Pradesh | 🟠 High |
| Manipur | 🟠 High |
| Sikkim | 🟡 Moderate |

---

## 👥 Target Users

This system is designed for **disaster management authorities**:
- NDMA (National Disaster Management Authority)
- State Disaster Management departments
- District Collectors in landslide-prone regions
- Emergency response teams

---

## 🎓 Project Info

- **Type:** Mini Project — B.Tech Data Science
- **Year:** 2026
- **Title:** DeepTerra: Explainable AI-Based Landslide Risk Prediction and Early Warning System

---

## 📜 License

This project is built for academic purposes.