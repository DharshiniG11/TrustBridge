"""
train_model.py (updated)
- Reads historical_data from Supabase
- Trains LinearRegression for delay_days
- Trains DecisionTreeClassifier for risk_level
- Evaluates models with up-to-date scikit-learn metrics
- Saves artifacts into ./models/
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, classification_report
from supabase import create_client, Client

# ----------------------------
# Configuration
# ----------------------------
FEATURE_COLS = ["progress_percent", "planned_duration", "labor_available", "supply_delays"]
TARGET_REG = "delay_days"
TARGET_CLF = "risk_level"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ----------------------------
# Supabase client
# ----------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ywsneogrjyrwvjgoeguh.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-service-role-key")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------
# Load historical data
# ----------------------------
print("📥 Fetching historical_data from Supabase...")
res = supabase.table("historical_data").select("*").execute()
hist = res.data
if not hist:
    raise SystemExit("❌ No historical_data rows found. Populate historical_data first.")

df = pd.DataFrame(hist)

# ----------------------------
# Basic cleaning / ensure features exist
# ----------------------------
for c in FEATURE_COLS:
    if c not in df.columns:
        raise SystemExit(f"❌ Required feature column missing in historical_data: {c}")

# Compute delay_days if not present
if TARGET_REG not in df.columns:
    if "actual_duration" in df.columns and "planned_duration" in df.columns:
        df[TARGET_REG] = df["actual_duration"].astype(float) - df["planned_duration"].astype(float)
        print("ℹ️ Computed delay_days = actual_duration - planned_duration")
    else:
        raise SystemExit("❌ delay_days missing and cannot be computed (no actual_duration)")

# Check classifier target
if TARGET_CLF not in df.columns:
    raise SystemExit("❌ risk_level missing in historical_data (needed for classifier)")

# Drop rows with missing feature values
df = df.dropna(subset=FEATURE_COLS + [TARGET_REG, TARGET_CLF]).reset_index(drop=True)

# Convert types
df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)
df[TARGET_REG] = df[TARGET_REG].astype(float)
df[TARGET_CLF] = df[TARGET_CLF].astype(str)

# ----------------------------
# Train/test split
# ----------------------------
X = df[FEATURE_COLS]
y_reg = df[TARGET_REG]
y_clf = df[TARGET_CLF]

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

# ----------------------------
# Train regression model
# ----------------------------
print("⚙️ Training regression model (LinearRegression)...")
reg = LinearRegression()
reg.fit(X_train, y_reg_train)

# Evaluate regression
y_reg_pred = reg.predict(X_test)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = root_mean_squared_error(y_reg_test, y_reg_pred)
r2 = reg.score(X_test, y_reg_test)

print(f"Regression results -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R^2: {r2:.3f}")

# Save regression model
with open(MODELS_DIR / "reg_model.pkl", "wb") as f:
    pickle.dump(reg, f)
print(f"✅ Saved regression model -> {MODELS_DIR/'reg_model.pkl'}")

# ----------------------------
# Train classifier model
# ----------------------------
print("⚙️ Training classifier (DecisionTreeClassifier) for risk_level...")
le = LabelEncoder()
y_clf_train_enc = le.fit_transform(y_clf_train)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_clf_train_enc)

# Evaluate classifier
y_clf_test_enc = le.transform(y_clf_test)
y_clf_pred_enc = clf.predict(X_test)
acc = accuracy_score(y_clf_test_enc, y_clf_pred_enc)
print(f"Classifier accuracy: {acc:.3f}")
print("Classification report:")
print(classification_report(y_clf_test_enc, y_clf_pred_enc, target_names=le.classes_.tolist()))

# Save classifier and encoder
with open(MODELS_DIR / "clf_model.pkl", "wb") as f:
    pickle.dump(clf, f)
with open(MODELS_DIR / "risk_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"✅ Saved classifier -> {MODELS_DIR/'clf_model.pkl'} and encoder -> {MODELS_DIR/'risk_encoder.pkl'}")

# ----------------------------
# Save feature columns
# ----------------------------
with open(MODELS_DIR / "feature_cols.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)
print(f"✅ Saved feature columns -> {MODELS_DIR/'feature_cols.pkl'}")

print("🎉 Training complete. You can now run ml_pipeline.py to make predictions.")
