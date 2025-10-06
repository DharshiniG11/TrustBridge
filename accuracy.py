import os
import pandas as pd
from supabase import create_client
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# -----------------------------
# Setup Supabase client
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL","https://ywsneogrjyrwvjgoeguh.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl3c25lb2dyanlyd3ZqZ29lZ3VoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzYwODY1NCwiZXhwIjoyMDczMTg0NjU0fQ.3is6-zQwNeFACjA4PYUqSV0Xj3Xu4WdjQNvmajR-vJo")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Fetch predictions
# -----------------------------
res_pred = supabase.table("predictions").select("*").execute()
pred_df = pd.DataFrame(res_pred.data)
if pred_df.empty:
    print("⚠️ No predictions found!")
    exit()
print(f"✅ Predictions table columns: {list(pred_df.columns)}")

# -----------------------------
# Fetch milestones
# -----------------------------
res_mile = supabase.table("milestones").select("*").execute()
mile_df = pd.DataFrame(res_mile.data)
if mile_df.empty:
    print("⚠️ No milestones found!")
    exit()
print(f"✅ Milestones table columns: {list(mile_df.columns)}")

# -----------------------------
# Merge predictions with milestones
# -----------------------------
df = pd.merge(pred_df, mile_df, left_on="milestone_id", right_on="id", suffixes=('_pred', '_mile'))

# -----------------------------
# Regression metrics: planned_duration vs predicted_handover
# -----------------------------
df['predicted_handover'] = pd.to_datetime(df['predicted_handover'])

# Compute predicted duration in days (from today as baseline)
df['predicted_duration'] = (df['predicted_handover'] - pd.Timestamp.today()).dt.days

# MAE / R² against planned_duration
mae = mean_absolute_error(df['planned_duration'], df['predicted_duration'])
r2 = r2_score(df['planned_duration'], df['predicted_duration'])
print(f"\n📊 Delay prediction metrics:")
print(f"- MAE: {mae:.2f} days")
print(f"- R²: {r2:.3f}")

# Normalize regression accuracy between 0 and 1
regression_accuracy = 1 - (mae / df['planned_duration'].mean())
regression_accuracy = max(0, min(1, regression_accuracy))

# -----------------------------
# Classification metrics: risk_level (if historical data exists)
# -----------------------------
res_hist = supabase.table("historical_data").select("id, project_id, milestone, risk_level").execute()
hist_df = pd.DataFrame(res_hist.data)

overall_accuracy = regression_accuracy  # default

if not hist_df.empty:
    # Merge predictions with historical data
    df_class = pd.merge(df, hist_df, left_on=['project_id', 'milestone'], 
                        right_on=['project_id', 'milestone'], suffixes=('_pred', '_actual'))
    if not df_class.empty:
        classification_accuracy = accuracy_score(df_class['risk_level_actual'], df_class['risk_level_pred'])
        overall_accuracy = (regression_accuracy + classification_accuracy) / 2
        print(f"\n📊 Risk prediction accuracy: {classification_accuracy*100:.2f}%")
    else:
        print("\n⚠️ No matching historical data to compute risk accuracy.")

print(f"\n✅ Overall prediction accuracy: {overall_accuracy*100:.2f}%")
print("\n✅ Script executed successfully.")