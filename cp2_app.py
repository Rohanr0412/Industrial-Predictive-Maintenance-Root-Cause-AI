import streamlit as st
import pandas as pd
import os
import joblib
import xgboost as xgb
# ============================================================
# CP2 — INDUSTRIAL PREDICTIVE MAINTENANCE & ROOT-CAUSE AI
# ============================================================

st.set_page_config(
    page_title="Industrial Predictive Maintenance AI",
    page_icon="🏭",
    layout="wide"
)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

st.title("🏭 Industrial Predictive Maintenance & Root-Cause AI")
st.markdown(
    "AI-powered machine failure prediction, component identification "
    "and maintenance recommendations."
)

st.divider()

# ------------------------------------------------------------
# Load deployment dataset
# ------------------------------------------------------------

DATA_PATH = "CP2_Deployment_Scoring_Dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df
# ------------------------------------------------------------
# Load component prediction model
# ------------------------------------------------------------

MODEL_FOLDER = "CP2_Final_Models"

@st.cache_resource
def load_component_model():

    model = xgb.XGBClassifier()
    model.load_model(
        os.path.join(
            MODEL_FOLDER,
            "component_xgboost.json"
        )
    )

    preprocessor = joblib.load(
        os.path.join(
            MODEL_FOLDER,
            "component_preprocessor.pkl"
        )
    )

    label_encoder = joblib.load(
        os.path.join(
            MODEL_FOLDER,
            "component_label_encoder.pkl"
        )
    )

    return model, preprocessor, label_encoder


try:
    component_model, component_preprocessor, component_label_encoder = (
        load_component_model()
    )

except Exception as e:
    st.error(
        f"Unable to load component prediction model: {e}"
    )
    st.stop()
try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load deployment dataset: {e}")
    st.stop()

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("Machine Selection")

machine_ids = sorted(df["machineID"].dropna().unique())

selected_machine = st.sidebar.selectbox(
    "Select Machine ID",
    machine_ids
)

# ------------------------------------------------------------
# Get selected machine data
# ------------------------------------------------------------

machine_data = df[
    df["machineID"] == selected_machine
].sort_values("datetime")

if machine_data.empty:
    st.warning("No data available for this machine.")
    st.stop()

# ------------------------------------------------------------
# Observation selection
# ------------------------------------------------------------

st.sidebar.divider()

observation_options = [
    "Latest Observation",
    "Highest Risk Observation"
]

observation_mode = st.sidebar.radio(
    "Observation",
    observation_options
)

if observation_mode == "Highest Risk Observation":
    selected_index = machine_data["failure_probability"].idxmax()
    latest = machine_data.loc[selected_index]
else:
    latest = machine_data.iloc[-1]

# ------------------------------------------------------------
# Prediction values
# ------------------------------------------------------------

failure_probability = float(
    latest.get("failure_probability", 0)
)

risk_level = str(
    latest.get("risk_level", "Unknown")
)

predicted_failure = int(
    latest.get("predicted_failure", 0)
)



# ------------------------------------------------------------
# Executive Alert Summary
# ------------------------------------------------------------

st.divider()

st.subheader("🚨 Executive Alert Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric(
        "Failure Probability",
        f"{failure_probability:.2%}"
    )

with summary_col2:
    st.metric(
        "Risk Level",
        risk_level
    )

with summary_col3:
    st.metric(
        "Predicted Failure",
        "YES" if predicted_failure == 1 else "NO"
    )

with summary_col4:
    st.metric(
        "Decision Threshold",
        "75%"
    )

if risk_level == "High":
    st.error(
        "🔴 HIGH-PRIORITY ALERT — Preventive inspection should be scheduled."
    )
elif risk_level == "Medium":
    st.warning(
        "🟡 MEDIUM-PRIORITY ALERT — Increase monitoring and review maintenance history."
    )
else:
    st.success(
        "🟢 NORMAL — Continue routine monitoring."
    )
# ------------------------------------------------------------
# Machine information
# ------------------------------------------------------------

st.subheader(f"Machine {selected_machine} — Current Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Machine Model",
        str(latest.get("model", "N/A"))
    )

with col2:
    st.metric(
        "Machine Age",
        f"{latest.get('machine_age_years', 0):.1f} years"
    )

with col3:
    st.metric(
        "Last Observation",
        latest["datetime"].strftime("%d %b %Y %H:%M")
    )

with col4:
    st.metric(
        "Days Since Maintenance",
        f"{latest.get('days_since_maintenance', 0):.1f}"
    )

# ------------------------------------------------------------
# Prediction section
# ------------------------------------------------------------

st.divider()

st.subheader("🚨 Failure Prediction")

failure_probability = float(
    latest.get("failure_probability", 0)
)

risk_level = str(
    latest.get("risk_level", "Unknown")
)

predicted_failure = int(
    latest.get("predicted_failure", 0)
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Failure Probability",
        f"{failure_probability:.2%}"
    )

with col2:
    st.metric(
        "Risk Level",
        risk_level
    )

with col3:
    st.metric(
        "Predicted Failure",
        "YES" if predicted_failure == 1 else "NO"
    )

# ------------------------------------------------------------
# Risk message
# ------------------------------------------------------------

if risk_level == "High":
    st.error(
        "🔴 HIGH RISK — Immediate preventive inspection recommended."
    )
elif risk_level == "Medium":
    st.warning(
        "🟡 MEDIUM RISK — Increase monitoring and review maintenance history."
    )
else:
    st.success(
        "🟢 LOW RISK — Continue routine monitoring."
    )

# ------------------------------------------------------------
# Component prediction
# ------------------------------------------------------------

# ------------------------------------------------------------
# Dynamic Component Prediction
# ------------------------------------------------------------

st.divider()

st.subheader("🔧 Predicted Failure Component")

# Only predict component when failure risk is meaningful
if failure_probability >= 0.75:

    try:

        # Get exactly the features used by the component model
        component_features = list(
            component_preprocessor.feature_names_in_
        )

        component_input = pd.DataFrame(
            [latest[component_features].to_dict()]
        )

        # Transform using training preprocessor
        component_input_processed = (
            component_preprocessor.transform(
                component_input
            )
        )

        # Predict component probabilities
        component_probabilities = (
            component_model.predict_proba(
                component_input_processed
            )[0]
        )

        # Convert encoded class back to component name
        predicted_class = component_probabilities.argmax()

        predicted_component = (
            component_label_encoder.inverse_transform(
                [predicted_class]
            )[0]
        )

        component_confidence = (
            component_probabilities[predicted_class]
        )

        # ----------------------------------------------------
        # Display main component prediction
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Likely Failure Component",
                predicted_component
            )

        with col2:
            st.metric(
                "Component Confidence",
                f"{component_confidence:.2%}"
            )

        # ----------------------------------------------------
        # Component probability breakdown
        # ----------------------------------------------------

        st.markdown(
            "#### Component Failure Probability"
        )

        component_chart = pd.DataFrame({
            "Component": [
                "Component 1",
                "Component 2",
                "Component 3",
                "Component 4"
            ],
            "Probability": (
                component_probabilities * 100
            )
        })

        st.bar_chart(
            component_chart.set_index("Component")
        )

    except Exception as e:

        st.warning(
            f"Component prediction could not be generated: {e}"
        )

else:

    st.info(
        "Component prediction is shown when the "
        "failure probability reaches the 0.75 decision threshold."
    )




    
# ------------------------------------------------------------
# SHAP / Explainability
# ------------------------------------------------------------

st.divider()

st.subheader("🔍 Top Model Factors")

factor_cols = [
    "top_factor_1",
    "top_factor_2",
    "top_factor_3"
]

available_factors = [
    latest[col]
    for col in factor_cols
    if col in latest.index and pd.notna(latest[col])
]

if available_factors:

    for i, factor in enumerate(available_factors, 1):
        st.write(f"**{i}.** {factor}")

else:
    st.info("Explainability factors are not available.")

# ------------------------------------------------------------
# Maintenance recommendation
# ------------------------------------------------------------

st.divider()

st.subheader("🛠️ Maintenance Recommendation")

recommendation = latest.get(
    "maintenance_recommendation",
    "No recommendation available."
)

if risk_level == "High":
    st.error(recommendation)
elif risk_level == "Medium":
    st.warning(recommendation)
else:
    st.success(recommendation)

# ------------------------------------------------------------
# Recent telemetry
# ------------------------------------------------------------

st.divider()

st.subheader("📈 Recent Telemetry")

telemetry_columns = [
    "datetime",
    "volt",
    "rotate",
    "pressure",
    "vibration"
]

available_telemetry = [
    col for col in telemetry_columns
    if col in machine_data.columns
]

recent_data = machine_data[
    available_telemetry
].tail(168)

st.line_chart(
    recent_data.set_index("datetime")[
        ["volt", "rotate", "pressure", "vibration"]
    ]
)

# ------------------------------------------------------------
# Machine history
# ------------------------------------------------------------

st.divider()

st.subheader("📋 Recent Machine History")

history_columns = [
    "datetime",
    "failure_probability",
    "risk_level",
    "predicted_component",
    "component_confidence",
    "maintenance_recommendation"
]

available_history = [
    col for col in history_columns
    if col in machine_data.columns
]

st.dataframe(
    machine_data[
        available_history
    ].tail(20),
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Industrial Predictive Maintenance & Root-Cause AI | "
    "XGBoost + SHAP + Multiclass Component Prediction + LSTM Benchmark"
)