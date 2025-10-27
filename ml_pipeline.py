import os
import pickle
import pandas as pd
from datetime import timezone
from dotenv import load_dotenv
from supabase import create_client, Client

# -----------------------------
# CONFIGURATION
# -----------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError("❌ SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# LOAD MODELS
# -----------------------------
print("📦 Loading trained models...")
with open("models/reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)
with open("models/clf_model.pkl", "rb") as f:
    clf_model = pickle.load(f)
with open("models/risk_encoder.pkl", "rb") as f:
    risk_encoder = pickle.load(f)
with open("models/feature_cols.pkl", "rb") as f:
    FEATURE_COLS = pickle.load(f)

# -----------------------------
# FETCH DATA
# -----------------------------
print("🔍 Fetching milestones from Supabase...")
res = supabase.table("milestones").select("*").execute()
mil_df = pd.DataFrame(res.data)

if mil_df.empty:
    raise SystemExit("⚠️ No milestones found!")

print(f"✅ Found {len(mil_df)} milestones")

# -----------------------------
# PREPARE FEATURES
# -----------------------------
for col in FEATURE_COLS:
    if col not in mil_df.columns:
        mil_df[col] = 0

mil_df[FEATURE_COLS] = mil_df[FEATURE_COLS].fillna({
    "progress_percent": 0,
    "planned_duration": 0,
    "labor_available": mil_df.get("labor_available", pd.Series([0])).median(),
    "supply_delays": 0,
    "concrete_m3": 0,
    "steel_tonnes": 0,
    "diesel_liters": 0
})

X = mil_df[FEATURE_COLS]

# -----------------------------
# MAKE PREDICTIONS
# -----------------------------
print("🤖 Generating predictions...")
mil_df["predicted_delay"] = reg_model.predict(X).round().astype(int)
risk_enc = clf_model.predict(X)
mil_df["predicted_risk_level"] = risk_encoder.inverse_transform(risk_enc)

today = pd.Timestamp.now(tz=timezone.utc).normalize()
mil_df["predicted_handover"] = today + pd.to_timedelta(
    mil_df["planned_duration"] + mil_df["predicted_delay"], unit="D"
)

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
# FEATURE 1: Resource Optimization
# -----------------------------
def optimize_resources(mil_df):
    """Suggest labor reallocation across projects."""
    print("\n🔧 Running resource optimization...")
    
    if "project_id" not in mil_df.columns:
        print("⚠️ Skipping: project_id column missing")
        return []
    
    project_summary = mil_df.groupby("project_id").agg({
        "predicted_delay": "sum",
        "labor_available": "mean",
        "predicted_risk_level": lambda x: (x == "High").sum()
    }).reset_index()
    
    overstaffed = project_summary[
        (project_summary["predicted_risk_level"] == 0) & 
        (project_summary["labor_available"] > project_summary["labor_available"].median())
    ]
    understaffed = project_summary[project_summary["predicted_risk_level"] > 1]
    
    recommendations = []
    for _, critical in understaffed.iterrows():
        for _, surplus in overstaffed.iterrows():
            if surplus["labor_available"] > 5:
                recommendations.append({
                    "action": "reallocate_labor",
                    "from_project_id": str(surplus["project_id"]),
                    "to_project_id": str(critical["project_id"]),
                    "workers": 5,
                    "reason": f"Prevent {int(critical['predicted_delay'])} day delay",
                    "created_at": pd.Timestamp.now(tz=timezone.utc).isoformat()
                })
                break
    
    if recommendations:
        try:
            supabase.table("resource_recommendations").insert(recommendations).execute()
            print(f"✅ Generated {len(recommendations)} resource recommendations")
        except Exception as e:
            print(f"⚠️ Failed to save: {e}")
    else:
        print("✅ No resource reallocation needed")
    
    return recommendations

# -----------------------------
# FEATURE 5: Task Resequencing
# -----------------------------
def resequence_tasks(mil_df):
    """Recalculate optimal task sequence when delays occur."""
    print("\n📊 Running task resequencing...")
    
    delayed_milestones = mil_df[mil_df["predicted_delay"] > 3].copy()
    
    if delayed_milestones.empty:
        print("✅ No significant delays - sequence optimal")
        return []
    
    all_alternatives = []
    
    for _, delayed in delayed_milestones.iterrows():
        try:
            deps_res = supabase.table("task_dependencies")\
                .select("dependent_task_id")\
                .eq("prerequisite_task_id", int(delayed["id"]))\
                .execute()
            
            if not deps_res.data:
                continue
            
            dependent_ids = [d["dependent_task_id"] for d in deps_res.data]
            dependent_tasks = mil_df[mil_df["id"].isin(dependent_ids)]
            
            for _, task in dependent_tasks.iterrows():
                wait_cost = int(delayed["predicted_delay"]) * 1000
                parallel_cost = 8000
                
                all_alternatives.append({
                    "milestone_id": int(delayed["id"]),
                    "dependent_task": task.get("milestone", "Unknown"),
                    "option_a": f"Wait {int(delayed['predicted_delay'])} days (${wait_cost})",
                    "option_b": f"Start parallel (${parallel_cost})",
                    "recommendation": "option_a" if wait_cost < parallel_cost else "option_b",
                    "created_at": pd.Timestamp.now(tz=timezone.utc).isoformat()
                })
        except Exception as e:
            print(f"⚠️ Error for milestone {delayed['id']}: {e}")
    
    if all_alternatives:
        try:
            supabase.table("sequencing_recommendations").insert(all_alternatives).execute()
            print(f"✅ Generated {len(all_alternatives)} sequencing options")
        except Exception as e:
            print(f"⚠️ Failed to save: {e}")
    else:
        print("✅ No task dependencies found (add some with task_dependencies table)")
    
    return all_alternatives

# -----------------------------
# FEATURE 7: Carbon Footprint
# -----------------------------
def calculate_carbon_footprint(mil_df):
    """Estimate CO2 emissions and suggest reductions."""
    print("\n🌱 Calculating carbon footprint...")
    
    CARBON_FACTORS = {
        "concrete": 410,
        "steel": 1850,
        "equipment_diesel": 2.68
    }
    
    mil_df["estimated_co2_kg"] = (
        mil_df.get("concrete_m3", pd.Series([0])).fillna(0) * CARBON_FACTORS["concrete"] +
        mil_df.get("steel_tonnes", pd.Series([0])).fillna(0) * CARBON_FACTORS["steel"] +
        mil_df.get("diesel_liters", pd.Series([0])).fillna(0) * CARBON_FACTORS["equipment_diesel"]
    )
    
    recommendations = []
    for _, row in mil_df[mil_df["estimated_co2_kg"] > 1000].iterrows():
        recommendations.append({
            "milestone_id": int(row["id"]),
            "current_co2_kg": float(row["estimated_co2_kg"]),
            "suggestion": "Switch to electric equipment (-15% CO2)",
            "potential_savings_kg": float(row["estimated_co2_kg"] * 0.15),
            "cost_impact": -500,
            "created_at": pd.Timestamp.now(tz=timezone.utc).isoformat()
        })
    
    if recommendations:
        try:
            supabase.table("sustainability_recommendations").insert(recommendations).execute()
            print(f"✅ Generated {len(recommendations)} sustainability recommendations")
        except Exception as e:
            print(f"⚠️ Failed to save: {e}")
    else:
        print("✅ All milestones below carbon threshold")
    
    return recommendations

# -----------------------------
# RUN ALL FEATURES
# -----------------------------
optimize_resources(mil_df)
resequence_tasks(mil_df)
calculate_carbon_footprint(mil_df)

# -----------------------------
# SAVE BASE PREDICTIONS
# -----------------------------
print("\n💾 Saving predictions to database...")
success_count = 0
for _, row in mil_df.iterrows():
    try:
        supabase.table("predictions").upsert(
            {
                "milestone_id": int(row["id"]),
                "predicted_handover": row["predicted_handover"].strftime("%Y-%m-%d"),
                "risk_level": row["predicted_risk_level"],
                "friendly_handover": row["friendly_handover"]
            },
            on_conflict="milestone_id"
        ).execute()
        success_count += 1
    except Exception as e:
        print(f"⚠️ Failed for milestone {row['id']}: {e}")

print(f"✅ Saved {success_count}/{len(mil_df)} predictions")

# -----------------------------
# SUMMARY
# -----------------------------
print("\n" + "="*60)
print("📈 PREDICTION SUMMARY")
print("="*60)
print(f"Total Milestones: {len(mil_df)}")
print(f"High Risk: {(mil_df['predicted_risk_level'] == 'High').sum()}")
print(f"Medium Risk: {(mil_df['predicted_risk_level'] == 'Medium').sum()}")
print(f"Low Risk: {(mil_df['predicted_risk_level'] == 'Low').sum()}")
print(f"Avg Predicted Delay: {mil_df['predicted_delay'].mean():.1f} days")
print(f"Total CO2 Estimate: {mil_df['estimated_co2_kg'].sum():,.0f} kg")
print("="*60)
print("\n✅ Pipeline completed successfully!")