import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client
import plotly.express as px
import subprocess

# -------------------------
# LOAD ENV VARIABLES
# -------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# PAGE SETUP
# -------------------------
st.set_page_config(page_title="TrustBridge AI Dashboard", layout="wide")
st.title("🏗️ TrustBridge – AI Project Intelligence Dashboard")

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
section = st.sidebar.radio(
    "Navigation",
    ["View Milestones", "View Predictions", "Run Pipeline", "Analytics Dashboard"]
)

# -------------------------
# SECTION 1 — View Milestones
# -------------------------
if section == "View Milestones":
    st.header("📌 All Milestones")
    try:
        res = supabase.table("milestones").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty:
            st.warning("No milestones found in database.")
        else:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load milestones: {e}")

# -------------------------
# SECTION 2 — View Predictions
# -------------------------
elif section == "View Predictions":
    st.header("🤖 AI Predictions")
    try:
        res = supabase.table("predictions").select("*").execute()
        df = pd.DataFrame(res.data)
        if df.empty:
            st.warning("No predictions found. Run the pipeline first.")
        else:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load predictions: {e}")

# -------------------------
# SECTION 3 — Run Pipeline
# -------------------------
elif section == "Run Pipeline":
    st.header("⚙️ Run ML Pipeline")
    if st.button("Run Pipeline Now"):
        st.info("Pipeline is running... please wait a few seconds.")
        try:
            # Run pipeline in Python subprocess to avoid Streamlit warnings
            subprocess.run(["python", "ml_pipeline.py"], check=True)
            st.success("Pipeline executed successfully ✅")
        except subprocess.CalledProcessError as e:
            st.error(f"Pipeline execution failed: {e}")

# -------------------------
# SECTION 4 — Analytics Dashboard
# -------------------------
elif section == "Analytics Dashboard":
    st.header("📊 Project Analytics & Summary")

    try:
        # Fetch predictions
        preds_res = supabase.table("predictions").select("*").execute()
        df_preds = pd.DataFrame(preds_res.data)

        if df_preds.empty:
            st.warning("No predictions available. Run the pipeline first.")
        else:
            # -------------------------
            # SUMMARY METRICS
            # -------------------------
            total_milestones = len(df_preds)
            high_risk = (df_preds['risk_level'] == 'High').sum()
            medium_risk = (df_preds['risk_level'] == 'Medium').sum()
            low_risk = (df_preds['risk_level'] == 'Low').sum()
            avg_delay = round(df_preds.get('predicted_delay', pd.Series([0])).mean(), 1)
            total_co2 = round(df_preds.get('estimated_co2_kg', pd.Series([0])).sum())

            st.subheader("📌 Prediction Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Milestones", total_milestones)
            col2.metric("High Risk", high_risk)
            col3.metric("Medium Risk", medium_risk)
            col4.metric("Low Risk", low_risk)
            col5.metric("Avg Predicted Delay (days)", avg_delay)

            st.markdown(f"**🌱 Total CO₂ Estimate:** {total_co2:,} kg")

            # -------------------------
            # RISK PIE CHART
            # -------------------------
            st.subheader("🔴 Risk Level Distribution")
            fig_risk = px.pie(
                df_preds,
                names="risk_level",
                title="Milestones by Risk Level",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_risk, width='stretch')

            # -------------------------
            # HANDOVER / DELAY HISTOGRAM
            # -------------------------
            st.subheader("⏱️ Predicted Handover Timeline")
            fig_delay = px.histogram(
                df_preds,
                x="predicted_handover",
                color="risk_level",
                title="Predicted Handover Dates by Risk Level"
            )
            st.plotly_chart(fig_delay, width='stretch')

            # -------------------------
            # CO2 FOOTPRINT BAR CHART
            # -------------------------
            if "estimated_co2_kg" in df_preds.columns:
                st.subheader("🌱 Carbon Footprint per Milestone")
                fig_co2 = px.bar(
                    df_preds,
                    x="milestone_id",
                    y="estimated_co2_kg",
                    color="risk_level",
                    title="Estimated CO₂ Emissions (kg)"
                )
                st.plotly_chart(fig_co2, width='stretch')

            # -------------------------
            # DOWNLOAD PREDICTIONS BUTTON
            # -------------------------
            csv = df_preds.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="trustbridge_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Failed to load analytics: {e}")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.write("📈 Streamlit Dashboard | TrustBridge Project")
