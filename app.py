
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. LOAD MODEL USING THE PATH OF THIS APP FILE
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "car_price_model.joblib"


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# Display useful diagnostics if the model cannot be loaded
if not MODEL_PATH.is_file():
    st.error("Model file not found.")
    st.write("Expected location:", str(MODEL_PATH))
    st.write("Files found in the app folder:")

    try:
        st.code("\n".join(p.name for p in APP_DIR.iterdir()))
    except Exception:
        st.write("Unable to list the app folder.")

    st.info(
        "Place car_price_model.joblib in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(
        f"Unable to load the model: {type(error).__name__}: {error}"
    )
    st.warning(
        "Ensure the model was saved successfully in Colab and "
        "that the local scikit-learn version is compatible."
    )
    st.stop()

# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.title("🚗 Indian Pre-Owned Car Price Predictor")

st.markdown(
    """
    Estimate the resale value of a pre-owned car in India using
    a trained machine-learning regression model.

    Enter the vehicle's specifications below to generate an estimate.
    """
)

st.success("Machine-learning model loaded successfully.")

with st.sidebar:
    st.header("About the Project")
    st.write(
        "This application uses a trained regression pipeline "
        "to estimate used-car prices from vehicle characteristics."
    )

    st.divider()
    st.caption("Project: Indian Pre-Owned Car Price Prediction")
    st.caption("Model file: car_price_model.joblib")

# ============================================================
# 4. VEHICLE DETAILS FORM
# ============================================================

st.subheader("Enter Car Details")

st.caption(
    "Use the same units and category labels as the training dataset. "
    "The example values below are illustrative."
)

with st.form("car_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        maker = st.text_input(
            "Car Maker",
            value="Maruti"
        )

        car_model = st.text_input(
            "Car Model",
            value=""
        )

        location = st.text_input(
            "Location",
            value=""
        )

        distance = st.number_input(
            "Distance Driven",
            min_value=0.0,
            value=30000.0,
            step=1000.0,
            help="Enter the distance using the dataset's original unit."
        )

        owner_type = st.text_input(
            "Owner Type",
            value=""
        )

    with col2:
        manufacture_year = st.number_input(
            "Manufacture Year",
            min_value=1980,
            max_value=2026,
            value=2020,
            step=1
        )

        age_of_car = st.number_input(
            "Age of Car",
            min_value=0.0,
            max_value=100.0,
            value=6.0,
            step=1.0
        )

        engine_displacement = st.number_input(
            "Engine Displacement",
            min_value=0.0,
            value=1200.0,
            step=100.0
        )

        engine_power = st.number_input(
            "Engine Power",
            min_value=0.0,
            value=80.0,
            step=5.0
        )

        audit_rating = st.number_input(
            "Vroom Audit Rating",
            min_value=0.0,
            value=5.0,
            step=1.0
        )

    with col3:
        body_type = st.text_input(
            "Body Type",
            value="Hatchback"
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

        door_count = st.number_input(
            "Door Count",
            min_value=0,
            value=4,
            step=1
        )

        seat_count = st.number_input(
            "Seat Count",
            min_value=0,
            value=5,
            step=1
        )

        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Other"]
        )

    submitted = st.form_submit_button(
        "🔍 Predict Car Price",
        use_container_width=True
    )

# ============================================================
# 5. PREPARE INPUT AND PREDICT
# ============================================================

if submitted:

    input_data = pd.DataFrame([{
        "Maker": maker,
        "model": car_model,
        "Location": location,
        "Distance": distance,
        "Owner Type": owner_type,
        "manufacture_year": manufacture_year,
        "Age of car": age_of_car,
        "engine_displacement": engine_displacement,
        "engine_power": engine_power,
        "body_type": body_type,
        "Vroom Audit Rating": audit_rating,
        "transmission": transmission,
        "door_count": door_count,
        "seat_count": seat_count,
        "fuel_type": fuel_type
    }])

    try:
        # IMPORTANT:
        # The saved pipeline performs the same preprocessing
        # used during training, followed by regression prediction.
        predicted_price = float(model.predict(input_data)[0])

        if not pd.notna(predicted_price):
            raise ValueError("The model returned an invalid prediction.")

        predicted_price = max(0.0, predicted_price)

        # ====================================================
        # 6. DISPLAY RESULTS
        # ====================================================

        st.divider()
        st.subheader("💰 Predicted Resale Price")

        st.metric(
            label="Estimated Car Price",
            value=f"₹ {predicted_price:,.0f}"
        )

        lower_price = predicted_price * 0.95
        upper_price = predicted_price * 1.05

        st.markdown("### Illustrative Price Range")

        range_col1, range_col2 = st.columns(2)

        with range_col1:
            st.metric(
                "Lower Estimate",
                f"₹ {lower_price:,.0f}"
            )

        with range_col2:
            st.metric(
                "Upper Estimate",
                f"₹ {upper_price:,.0f}"
            )

        st.caption(
            "The range is an illustrative ±5% band around the "
            "prediction, not a statistically validated confidence interval."
        )

        with st.expander("View entered vehicle details"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

        st.info(
            "The estimate depends on the quality of the training data "
            "and model performance. Actual resale prices may vary."
        )

    except Exception as error:
        st.error(
            f"Prediction failed: {type(error).__name__}: {error}"
        )

        st.write(
            "Check that the input feature names and data types match "
            "the features used when the model was trained."
        )

        st.write("Features supplied to the model:")
        st.code(", ".join(input_data.columns))

# ============================================================
# 7. FOOTER
# ============================================================

st.divider()

st.caption(
    "Indian Pre-Owned Car Price Prediction | "
    "Machine Learning + Streamlit"
)
