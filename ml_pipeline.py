"""
ml_pipeline.py (Updated & Fixed)
- Fetches live milestones from Supabase
- Handles missing values in features
- Predicts delay_days and risk_level using trained models
- Calculates predicted_handover date
- Adds friendly_handover string for dashboard
- Upserts predictions into Supabase
"""

import os
import pickle
import pandas as pd
from datetime import date
from dotenv import load_dotenv
from supabase import create_client, Client

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ywsneogrjyrwvjgoeguh.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-service-role-key")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Load models and feature columns
# -----------------------------
with open("models/reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("models/clf_model.pkl", "rb") as f:
    clf_model = pickle.load(f)

with open("models/risk_encoder.pkl", "rb") as f:
    risk_encoder = pickle.load(f)

with open("models/feature_cols.pkl", "rb") as f:
    FEATURE_COLS = pickle.load(f)

# -----------------------------
# Helper: Friendly string for dashboard
# -----------------------------
def friendly_handover(days: int) -> str:
    if days < 7:
        return "Within a week"
    elif days < 30:
        return f"In ~{days} days"
    elif days < 60:
        return f"In ~{days//7} weeks"
    else:
        return f"In ~{days//30} months"

# -----------------------------
# Main function
# -----------------------------
def predict_and_update():
    print("📥 Fetching live milestones from Supabase...")
    res = supabase.table("milestones").select("*").execute()
    milestones = res.data
    if not milestones:
        print("⚠️ No milestones found!")
        return

    mil_df = pd.DataFrame(milestones)

    # Ensure all FEATURE_COLS exist and fill NaNs
    for col in FEATURE_COLS:
        if col not in mil_df.columns:
            mil_df[col] = 0

    # Fill missing values with defaults
    mil_df[FEATURE_COLS] = mil_df[FEATURE_COLS].fillna({
        "progress_percent": 0,
        "planned_duration": 0,
        "labor_available": mil_df["labor_available"].median() if "labor_available" in mil_df else 0,
        "supply_delays": 0
    })

    X = mil_df[FEATURE_COLS]

    print("🤖 Running predictions...")
    # Predict numeric delay_days
    mil_df["predicted_delay"] = reg_model.predict(X).round().astype(int)

    # Predict risk_level
    risk_enc = clf_model.predict(X)
    mil_df["predicted_risk_level"] = risk_encoder.inverse_transform(risk_enc)

    # Compute predicted handover date (vectorized)
    today = pd.Timestamp.today().normalize()  # Timestamp at 00:00
    mil_df["predicted_handover"] = today + pd.to_timedelta(
        mil_df["planned_duration"] + mil_df["predicted_delay"], unit="D"
    )

    # Friendly handover string
    mil_df["friendly_handover"] = mil_df["predicted_delay"].apply(friendly_handover)

    # Upsert predictions to Supabase
    for _, row in mil_df.iterrows():
        supabase.table("predictions").upsert({
            "milestone_id": row["id"],
            "predicted_handover": row["predicted_handover"].strftime("%Y-%m-%d"),
            "risk_level": row["predicted_risk_level"],
            "friendly_handover": row["friendly_handover"]
        }).execute()
        print(f"✅ Milestone {row['milestone']} updated: Handover {row['friendly_handover']}, Risk {row['predicted_risk_level']}")

# -----------------------------
if __name__ == "__main__":
    predict_and_update()
