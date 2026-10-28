
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1. Page configuration
# -------------------------------
st.set_page_config(
    page_title="PredictivePulse",
    page_icon="⚙️",
    layout="wide"
)

# -------------------------------
# 2. Load saved model and columns
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, model_columns

try:
    model, model_columns = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# -------------------------------
# 3. Dashboard header
# -------------------------------
st.title("⚙️ PredictivePulse")
st.subheader("Smart Machine Failure Prediction")

st.write(
    "Enter the machine operating parameters to predict "
    "whether the machine may experience a failure."
)

st.divider()

# -------------------------------
# 4. Machine input section
# -------------------------------
st.header("Machine Parameters")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=0,
        max_value=10000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=300,
        value=100
    )

st.divider()

# -------------------------------
# 5. Predict machine failure
# -------------------------------
if st.button("Predict Machine Status", type="primary"):

    # Build input using training column names
    input_data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M")
    }

    input_df = pd.DataFrame([input_data])

    # Match the model's exact feature order
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.error("⚠️ Potential Machine Failure Detected")
            st.write(
                "The model predicts a machine failure. "
                "Inspect the equipment and follow your "
                "maintenance procedures."
            )
        else:
            st.success("✅ No Machine Failure Predicted")
            st.write(
                "The model predicts no failure for the "
                "provided operating parameters."
            )

        # Show probability when supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                failure_probability = probabilities[
                    classes.index(1)
                ]

                st.metric(
                    "Estimated Failure Probability",
                    f"{failure_probability:.2%}"
                )

                st.caption(
                    "This is the model's estimated probability, "
                    "not a guarantee of machine condition."
                )

        with st.expander("View submitted machine parameters"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("PredictivePulse | Machine Learning Project")
