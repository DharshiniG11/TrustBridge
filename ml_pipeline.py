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
# Load trained models
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
# Fetch live milestones
# -----------------------------
res = supabase.table("milestones").select("*").execute()
mil_df = pd.DataFrame(res.data)

if mil_df.empty:
    raise SystemExit("⚠️ No milestones found!")

# -----------------------------
# Fill missing feature values
# -----------------------------
for col in FEATURE_COLS:
    if col not in mil_df.columns:
        mil_df[col] = 0

mil_df[FEATURE_COLS] = mil_df[FEATURE_COLS].fillna({
    "progress_percent": 0,
    "planned_duration": 0,
    "labor_available": mil_df["labor_available"].median() if "labor_available" in mil_df else 0,
    "supply_delays": 0
})

X = mil_df[FEATURE_COLS]

# -----------------------------
# Old predictions (simulate current dashboard)
# -----------------------------
mil_df["old_predicted_handover"] = mil_df["planned_duration"].apply(
    lambda d: pd.Timestamp.today() + pd.to_timedelta(d, unit="D")
)
mil_df["old_risk_level"] = "Medium"

# -----------------------------
# New predictions (improved ML)
# -----------------------------
mil_df["predicted_delay"] = reg_model.predict(X).round().astype(int)
risk_enc = clf_model.predict(X)
mil_df["predicted_risk_level"] = risk_encoder.inverse_transform(risk_enc)

today = pd.Timestamp.today().normalize()
mil_df["predicted_handover"] = today + pd.to_timedelta(
    mil_df["planned_duration"] + mil_df["predicted_delay"], unit="D"
)

# -----------------------------
# Friendly handover string
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

mil_df["friendly_handover"] = mil_df["predicted_delay"].apply(friendly_handover)

# -----------------------------
# Show comparison before vs after
# -----------------------------
# -----------------------------
# Show before vs after with project_id
# -----------------------------
comparison_cols = [
    "project_id", "milestone", "planned_duration",
    "old_predicted_handover", "predicted_handover",
    "old_risk_level", "predicted_risk_level",
    "friendly_handover"
]
print(mil_df[comparison_cols].to_string(index=False))


# -----------------------------
# Upsert predictions to Supabase
# -----------------------------
for _, row in mil_df.iterrows():
    supabase.table("predictions").upsert(
        {
            "milestone_id": row["id"],
            "predicted_handover": row["predicted_handover"].strftime("%Y-%m-%d"),
            "risk_level": row["predicted_risk_level"],
            "friendly_handover": row["friendly_handover"]
        },
        on_conflict="milestone_id"  # ← prevents duplicates, updates existing rows
    ).execute()

print("✅ Predictions upserted successfully.")
