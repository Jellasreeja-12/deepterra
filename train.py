# ============================================================
#   DeepTerra — train.py
#   Explainable AI-Based Landslide Risk Prediction
#   ANN + SHAP | TensorFlow/Keras
# ============================================================

import warnings
import os
import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

print("=" * 55)
print("  DeepTerra — ANN + Explainable AI Training")
print("=" * 55)

# ── CREATE MODEL FOLDER (IMPORTANT FIX) ──
os.makedirs("model", exist_ok=True)

# ── STEP 1: LOAD DATA ──────────────────────────────────────
print("\n[1/8] Loading dataset...")
df = pd.read_csv("regenerated_landslide_risk_dataset.csv")

print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
print(
    f"   Label distribution:\n{df['Landslide Risk Prediction'].value_counts().to_string()}")

# ── STEP 2: PREPROCESS ────────────────────────────────────
print("\n[2/8] Preprocessing...")

df['label'] = df['Landslide Risk Prediction'].apply(
    lambda x: 0 if x == 'Low' else 1
)

feature_cols = [
    'Temperature (°C)', 'Humidity (%)',
    'Precipitation (mm)', 'Soil Moisture (%)', 'Elevation (m)'
]

X = df[feature_cols]
y = df['label']

print(f"   Safe (0): {(y==0).sum()} | Landslide Risk (1): {(y==1).sum()}")

# ── STEP 3: BALANCE ───────────────────────────────────────
print("\n[3/8] Balancing classes...")

majority = df[df['label'] == 0]
minority = df[df['label'] == 1]

TARGET = min(2000, len(majority))

majority_down = majority.sample(TARGET, random_state=42)
minority_up = resample(minority, replace=True,
                       n_samples=TARGET, random_state=42)

df_bal = pd.concat([majority_down, minority_up]).sample(
    frac=1, random_state=42).reset_index(drop=True)
print(
    f"   After balancing — Safe: {(df_bal['label']==0).sum()} | Risk: {(df_bal['label']==1).sum()}")

X_bal = df_bal[feature_cols].values
y_bal = df_bal['label'].values

print(
    f"   After balancing — Safe: {(y_bal==0).sum()} | Risk: {(y_bal==1).sum()}")

print(
    f"   After balancing — Safe: {(y_bal==0).sum()} | Risk: {(y_bal==1).sum()}")

# ── STEP 4: SCALE ─────────────────────────────────────────
print("\n[4/8] Scaling features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_bal)

# ── STEP 5: SPLIT ─────────────────────────────────────────
print("\n[5/8] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_bal,
    test_size=0.2,
    random_state=42,
    stratify=y_bal
)

print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── STEP 6: MODEL ─────────────────────────────────────────
print("\n[6/8] Building ANN model...")

model = Sequential([
    Dense(128, activation='relu', input_shape=(len(feature_cols),)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(16, activation='relu'),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ── STEP 7: TRAIN ─────────────────────────────────────────
print("\n[7/8] Training...")

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ── STEP 8: EVALUATE ──────────────────────────────────────
print("\n[8/8] Evaluating...")

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*40)
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print("="*40)

# ── CONFUSION MATRIX ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:")
print(f"  True Negatives  (correctly Safe):     {cm[0][0]}")
print(f"  False Positives (false alarm):         {cm[0][1]}")
print(f"  False Negatives (missed risk!):        {cm[1][0]}")
print(f"  True Positives  (correctly detected):  {cm[1][1]}")
# ── SAVE CONFUSION MATRIX IMAGE ───────────────────────────
plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

ticks = np.arange(2)
plt.xticks(ticks, ['Safe', 'Risk'])
plt.yticks(ticks, ['Safe', 'Risk'])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=120)
plt.close()

print("Confusion matrix saved")

# ── SHAP (FIXED SAFE VERSION) ─────────────────────────────
print("\nGenerating SHAP explanations...")

explainer = shap.KernelExplainer(
    lambda x: model.predict(x, verbose=0).flatten(),
    X_train[:100]
)

X_test_sample = X_test[:100]

shap_values = explainer.shap_values(X_test_sample, nsamples=100)

# FIX: safe conversion (Streamlit-safe)
shap_array = np.array(shap_values)
mean_shap = np.abs(shap_array).mean(axis=0).flatten()

shap_dict = dict(zip(feature_cols, mean_shap.tolist()))

# ── SAVE SHAP PLOT ───────────────────────────────────────
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_sample,
    feature_names=feature_cols,
    show=False,
    plot_type="bar"
)

plt.tight_layout()
plt.savefig("model/shap_summary.png", dpi=120)
plt.close()

print("SHAP saved")

# ── SAVE ARTIFACTS ────────────────────────────────────────
model.save("model/deepterra_ann.keras")

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/metadata.pkl", "wb") as f:
    pickle.dump({
        "feature_cols": feature_cols,
        "shap_dict": shap_dict,
        "accuracy": round(acc * 100, 2),
        "f1": round(f1 * 100, 2)
    }, f)

print("\nDONE. Ready for Streamlit deployment.")
