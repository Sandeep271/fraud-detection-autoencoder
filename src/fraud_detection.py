# src/fraud_detection.py

import os
import pandas as pd
import numpy as np
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load Dataset with Absolute Path ---
print(f"Current working directory: {os.getcwd()}")

# Construct data path
data_path = os.path.join(os.getcwd(), "data", "creditcard.csv")
print(f"Looking for file at: {data_path}")
print(f"File exists: {os.path.exists(data_path)}")

# Load dataset
df = pd.read_csv(data_path)
print(f"Dataset loaded successfully with shape: {df.shape}")

# --- Step 2: Preprocessing ---
print("🔄 Scaling features...")
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("🔄 Filtering normal transactions for training...")
X_train = X_scaled[y == 0]  # Only normal transactions (Class = 0)

# --- Step 3: Initialize AutoEncoder Model ---
print("⚙️ Initializing AutoEncoder...")
autoencoder = AutoEncoder()  # Use default parameters

# --- Step 4: Train the Model ---
print("🚀 Training AutoEncoder on normal data...")
autoencoder.fit(X_train)

# --- Step 5: Predict ---
print("🔍 Predicting anomalies...")
y_pred = autoencoder.predict(X_scaled)               # 0 = normal, 1 = anomaly
y_scores = autoencoder.decision_function(X_scaled)   # anomaly score

# --- Step 6: Evaluation Metrics ---
print("📊 Evaluation Results:")
print(classification_report(y, y_pred, digits=4))

# --- Step 7: Confusion Matrix Plot ---
print("📈 Generating confusion matrix...")
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# Save confusion matrix
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

conf_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(conf_matrix_path)
print(f"✅ Confusion matrix saved to {conf_matrix_path}")
plt.close()

# --- Step 8: Precision-Recall Curve ---
print("📉 Generating precision-recall curve...")
precision, recall, _ = precision_recall_curve(y, y_scores)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

pr_curve_path = os.path.join(output_dir, "precision_recall_curve.png")
plt.savefig(pr_curve_path)
print(f"✅ Precision-Recall curve saved to {pr_curve_path}")
plt.close()

# --- Step 9: Save Predictions to CSV ---
print("💾 Saving prediction results to CSV...")
df["prediction"] = y_pred
df["anomaly_score"] = y_scores

results_csv = os.path.join(output_dir, "prediction_results.csv")
df.to_csv(results_csv, index=False)
print(f"✅ Predictions saved to {results_csv}")

print("🎉 All steps completed successfully.")
