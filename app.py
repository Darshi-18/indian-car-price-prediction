
import streamlit as st
import pandas as pd
import joblib
import gdown
import os
from pathlib import Path

# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "car_price_model_compressed.joblib"

# Paste your Google Drive file ID here.
GOOGLE_DRIVE_FILE_ID = "1YDqqIyv8gkE8lIQzQLOE9qA4NdoeHch2"

# ============================================================
# 2. CUSTOM UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #17365d;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 16px;
        color: #526579;
        text-align: center;
        margin-bottom: 25px;
    }

    .result-box {
        background-color: #e8f5e9;
        padding: 25px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid #b7dfbb;
    }

    .result-label {
        color: #245b2a;
        font-size: 18px;
        font-weight: 600;
    }

    .result-price {
        color: #17652b;
        font-size: 36px;
        font-weight: 800;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 700;
        min-height: 45px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3. DOWNLOAD AND LOAD THE TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Downloads the model from Google Drive if necessary,
    then loads the trained scikit-learn pipeline.
    """

    if not MODEL_PATH.exists():
        if (
            not GOOGLE_DRIVE_FILE_ID
            or GOOGLE_DRIVE_FILE_ID == "1YDqqIyv8gkE8lIQzQLOE9qA4NdoeHch2"
        ):
            raise ValueError(
                "Please add your Google Drive file ID "
                "in the GOOGLE_DRIVE_FILE_ID variable."
            )

        downloaded_file = gdown.download(
            id=GOOGLE_DRIVE_FILE_ID,
            output=str(MODEL_PATH),
            quiet=False
        )

        if (
            downloaded_file is None
            or not MODEL_PATH.exists()
            or MODEL_PATH.stat().st_size == 0
        ):
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()

            raise RuntimeError(
                "Model download failed. Check the Drive file ID, "
                "sharing permissions, and download quota."
            )

    # Load the model trained in your Colab notebook.
    return joblib.load(MODEL_PATH)


# ============================================================
# 4. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚘 Indian Pre-Owned Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Estimate a used car’s price using its specifications, '
    'age, condition-related attributes, and location.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:
    st.header("About the Project")

    st.write(
        "This application uses a machine-learning model trained "
        "on Indian pre-owned car market data."
    )

    st.markdown("### Features considered")
    st.write(
        "- Manufacturer and model\n"
        "- Location and ownership\n"
        "- Manufacturing year and car age\n"
        "- Engine specifications\n"
        "- Fuel and transmission types\n"
        "- Body type and seating capacity\n"
        "- Vroom Audit Rating"
    )

    st.divider()

    st.caption(
        "Prediction results are estimates, not guaranteed "
        "market offers."
    )

# ============================================================
# 6. CAR INPUT FORM
# ============================================================

st.subheader("Enter Car Details")
st.write("Fill in the information below and click Predict Price.")

with st.form("car_prediction_form"):

    st.markdown("### 🚗 Basic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        maker = st.selectbox(
            "Car Manufacturer",
            [
                "Maruti Suzuki", "Hyundai", "Honda", "Tata",
                "Mahindra", "Toyota", "Kia", "Renault",
                "Volkswagen", "Skoda", "Ford", "BMW",
                "Mercedes-Benz", "Audi", "Nissan", "MG",
                "Jeep", "Datsun", "Chevrolet", "Other"
            ]
        )

    with col2:
        model_name = st.text_input(
            "Car Model",
            value="Swift",
            help="Enter the model name as represented in the training data."
        ).strip()

    with col3:
        location = st.text_input(
            "City / Location",
            value="Delhi"
        ).strip()

    col4, col5, col6 = st.columns(3)

    with col4:
        manufacture_year = st.number_input(
            "Manufacturing Year",
            min_value=1980,
            max_value=2026,
            value=2020,
            step=1
        )

    with col5:
        current_year = 2026
        default_age = max(0, current_year - int(manufacture_year))

        age_of_car = st.number_input(
            "Age of Car (years)",
            min_value=0,
            max_value=60,
            value=default_age,
            step=1
        )

    with col6:
        distance = st.number_input(
            "Distance Driven (km)",
            min_value=0,
            max_value=2_000_000,
            value=35000,
            step=1000
        )

    st.markdown("### ⚙️ Engine and Performance")

    col7, col8, col9 = st.columns(3)

    with col7:
        engine_displacement = st.number_input(
            "Engine Displacement (cc)",
            min_value=0,
            max_value=10000,
            value=1197,
            step=1
        )

    with col8:
        engine_power = st.number_input(
            "Engine Power (bhp)",
            min_value=0.0,
            max_value=2000.0,
            value=82.0,
            step=1.0
        )

    with col9:
        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"]
        )

    st.markdown("### 🛞 Body and Transmission")

    col10, col11, col12 = st.columns(3)

    with col10:
        body_type = st.selectbox(
            "Body Type",
            [
                "Hatchback", "Sedan", "SUV", "MUV",
                "Coupe", "Convertible", "Minivan",
                "Pickup Truck", "Wagon", "Other"
            ]
        )

    with col11:
        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

    with col12:
        owner_type = st.selectbox(
            "Owner Type",
            [
                "First", "Second", "Third", "Fourth & Above",
                "First Owner", "Second Owner", "Third Owner",
                "Fourth & Above Owner"
            ]
        )

    col13, col14, col15 = st.columns(3)

    with col13:
        door_count = st.selectbox(
            "Number of Doors",
            [2, 3, 4, 5, 6]
        )

    with col14:
        seat_count = st.selectbox(
            "Seating Capacity",
            [2, 4, 5, 6, 7, 8, 9, 10]
        )

    with col15:
        audit_rating = st.number_input(
            "Vroom Audit Rating",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.5
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Car Price",
        type="primary"
    )

# ============================================================
# 7. VALIDATE INPUTS AND MAKE A PREDICTION
# ============================================================

if submitted:

    if not model_name:
        st.error("Please enter the car model.")
        st.stop()

    if not location:
        st.error("Please enter the car's location.")
        st.stop()

    # Keep these column names aligned with the cleaned training data.
    input_data = pd.DataFrame([{
        "Maker": maker,
        "model": model_name,
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

    st.subheader("Prediction Result")

    try:
        with st.spinner("Loading model and predicting price..."):
            model = load_model()
            prediction = model.predict(input_data)
            predicted_price = float(prediction[0])

        if predicted_price < 0:
            st.warning(
                "The model returned a negative price. "
                "Please check the model and input data."
            )
        else:
            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">
                        Estimated Pre-Owned Car Price
                    </div>
                    <div class="result-price">
                        ₹ {predicted_price:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "This is a machine-learning estimate. Actual selling "
                "prices may vary with condition, documentation, "
                "negotiation, and market demand."
            )

            with st.expander("View submitted car details"):
                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as error:
        st.error("Unable to generate the prediction.")

        st.write(
            "Please check that the model file is accessible and that "
            "the app uses the same scikit-learn version and feature "
            "schema used during training."
        )

        st.code(str(error))
