# ============================================================
#   DeepTerra — train.py (Streamlit Cloud Compatible)
#   ANN via scikit-learn MLPClassifier + SHAP
# ============================================================

import warnings
import os
import pickle
import shap
from sklearn.utils import resample
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


print("=" * 55)
print("  DeepTerra — ANN + Explainable AI Training")
print("  (Streamlit Cloud Compatible Version)")
print("=" * 55)

# ── LOAD ──
print("\n[1/7] Loading dataset...")
df = pd.read_csv("regenerated_landslide_risk_dataset.csv")
print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
print(
    f"   Label distribution:\n{df['Landslide Risk Prediction'].value_counts().to_string()}")

# ── PREPROCESS ──
print("\n[2/7] Preprocessing...")
df['label'] = df['Landslide Risk Prediction'].apply(
    lambda x: 0 if x == 'Low' else 1
)
feature_cols = [
    'Temperature (°C)', 'Humidity (%)',
    'Precipitation (mm)', 'Soil Moisture (%)', 'Elevation (m)'
]
print(
    f"   Safe (0): {(df['label']==0).sum()} | Risk (1): {(df['label']==1).sum()}")

# ── BALANCE ──
print("\n[3/7] Balancing classes...")
majority = df[df['label'] == 0]
minority = df[df['label'] == 1]
TARGET = min(2000, len(majority))
maj_down = majority.sample(TARGET, random_state=42)
min_up = resample(minority, replace=True, n_samples=TARGET, random_state=42)
df_bal = pd.concat([maj_down, min_up]).sample(
    frac=1, random_state=42).reset_index(drop=True)
print(
    f"   Balanced — Safe: {(df_bal['label']==0).sum()} | Risk: {(df_bal['label']==1).sum()}")

X = df_bal[feature_cols].values
y = df_bal['label'].values

# ── SCALE ──
print("\n[4/7] Scaling...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── SPLIT ──
print("\n[5/7] Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── BUILD ANN ──
print("\n[6/7] Training ANN (MLPClassifier)...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)
model.fit(X_train, y_train)

# ── EVALUATE ──
print("\n[7/7] Evaluating...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  ✅ Accuracy  : {acc*100:.2f}%")
print(f"  ✅ Precision : {prec*100:.2f}%")
print(f"  ✅ Recall    : {rec*100:.2f}%")
print(f"  ✅ F1 Score  : {f1*100:.2f}%")
print(f"{'='*40}")
print(classification_report(y_test, y_pred,
      target_names=['Safe', 'Landslide Risk']))

cm = confusion_matrix(y_test, y_pred)
print(f"  True Negatives  : {cm[0][0]}")
print(f"  False Positives : {cm[0][1]}")
print(f"  False Negatives : {cm[1][0]}")
print(f"  True Positives  : {cm[1][1]}")

# ── SHAP ──
print("\nGenerating SHAP explanations...")
os.makedirs("model", exist_ok=True)
explainer = shap.KernelExplainer(model.predict_proba, X_train[:100])
shap_values = explainer.shap_values(X_test[:100], nsamples=100)

# For binary classification shap_values is a list — take class 1
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(8, 5))
shap.summary_plot(
    sv, X_test[:100], feature_names=feature_cols, show=False, plot_type='bar')
plt.title("Feature Importance — SHAP Values")
plt.tight_layout()
plt.savefig("model/shap_summary.png", dpi=120, bbox_inches='tight')
plt.close()
print("   SHAP plot saved → model/shap_summary.png")

mean_shap = np.abs(sv).mean(axis=0)
shap_dict = dict(zip(feature_cols, mean_shap.tolist()))

# ── SAVE ──
with open("model/deepterra_model.pkl", "wb") as f:
    pickle.dump(model,  f)
with open("model/scaler.pkl",          "wb") as f:
    pickle.dump(scaler, f)
with open("model/metadata.pkl",        "wb") as f:
    pickle.dump({
        'feature_cols': feature_cols,
        'shap_dict':    shap_dict,
        'accuracy':     round(acc * 100, 2),
        'f1':           round(f1 * 100, 2),
    }, f)

print("\n✅ All files saved to model/")
print("➡️  Run: streamlit run app.py")
