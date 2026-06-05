# ============================================================
#   DeepTerra — train.py
#   Explainable AI-Based Landslide Risk Prediction
#   ANN + SHAP | TensorFlow/Keras
# ============================================================

import warnings
import os
import pickle
import shap
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.utils import resample
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


print("=" * 55)
print("  DeepTerra — ANN + Explainable AI Training")
print("=" * 55)

# ── STEP 1: LOAD DATA ──────────────────────────────────────
print("\n[1/8] Loading dataset...")
df = pd.read_csv("regenerated_landslide_risk_dataset.csv")
print(f"   Rows: {len(df)} | Columns: {len(df.columns)}")
print(
    f"   Label distribution:\n{df['Landslide Risk Prediction'].value_counts().to_string()}")

# ── STEP 2: PREPROCESS ────────────────────────────────────
print("\n[2/8] Preprocessing...")

# Convert to binary: Low=0 (Safe), rest=1 (Landslide Risk)
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

# ── STEP 3: BALANCE CLASSES ───────────────────────────────
print("\n[3/8] Balancing classes...")
df_balanced = df.copy()
majority = df_balanced[df_balanced['label'] == 0]
minority = df_balanced[df_balanced['label'] == 1]

# Oversample minority to match majority (capped at 2000 each for speed)
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

# ── STEP 4: SCALE ─────────────────────────────────────────
print("\n[4/8] Scaling features...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_bal)

# ── STEP 5: TRAIN/TEST SPLIT ──────────────────────────────
print("\n[5/8] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ── STEP 6: BUILD ANN ─────────────────────────────────────
print("\n[6/8] Building ANN model...")
model = Sequential([
    # Input layer
    Dense(128, activation='relu', input_shape=(len(feature_cols),)),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden layer 1
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden layer 2
    Dense(32, activation='relu'),
    Dropout(0.2),

    # Hidden layer 3
    Dense(16, activation='relu'),

    # Output layer (binary)
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── STEP 7: TRAIN ─────────────────────────────────────────
print("\n[7/8] Training ANN...")
early_stop = EarlyStopping(
    monitor='val_accuracy', patience=10,
    restore_best_weights=True
)

history = model.fit(
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

print(f"\n{'='*40}")
print(f"  ✅ Accuracy  : {acc*100:.2f}%")
print(f"  ✅ Precision : {prec*100:.2f}%")
print(f"  ✅ Recall    : {rec*100:.2f}%")
print(f"  ✅ F1 Score  : {f1*100:.2f}%")
print(f"{'='*40}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Safe', 'Landslide Risk']))

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

tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Safe', 'Risk'])
plt.yticks(tick_marks, ['Safe', 'Risk'])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha='center',
            va='center'
        )

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png",
    dpi=120,
    bbox_inches='tight'
)
plt.close()

print("   Confusion matrix saved → model/confusion_matrix.png")

# ── SHAP EXPLAINABILITY ───────────────────────────────────
print("\nGenerating SHAP explanations...")
os.makedirs("model", exist_ok=True)

# Use a background sample for SHAP
background = X_train[:100]
explainer = shap.KernelExplainer(
    lambda x: model.predict(x, verbose=0).flatten(),
    background
)

# Explain test sample
X_test_sample = X_test[:100]
shap_values = explainer.shap_values(X_test_sample, nsamples=100)

# Save SHAP summary plot
plt.figure(figsize=(8, 5))
shap.summary_plot(
    shap_values, X_test_sample,
    feature_names=feature_cols,
    show=False, plot_type='bar'
)
plt.title("Feature Importance — SHAP Values")
plt.tight_layout()
plt.savefig("model/shap_summary.png", dpi=120, bbox_inches='tight')
plt.close()
print("   SHAP summary plot saved → model/shap_summary.png")

# Save mean SHAP values for app
mean_shap = np.abs(shap_values).mean(axis=0)
shap_dict = dict(zip(feature_cols, mean_shap.tolist()))
print(f"   Feature importance: {shap_dict}")

# ── SAVE MODEL & ARTIFACTS ────────────────────────────────
model.save("model/deepterra_ann.keras")

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/metadata.pkl", "wb") as f:
    pickle.dump({
        'feature_cols': feature_cols,
        'shap_dict':    shap_dict,
        'accuracy':     round(acc * 100, 2),
        'f1':           round(f1 * 100, 2),
    }, f)

print("\n✅ All files saved to model/")
print("➡️  Run: streamlit run app.py")
