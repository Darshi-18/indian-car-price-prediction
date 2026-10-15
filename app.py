
import streamlit as st
import pandas as pd
import joblib
import gdown
from pathlib import Path

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Indian Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "car_price_model.joblib"

# Your Google Drive model file ID
GOOGLE_DRIVE_FILE_ID = "1YDqqIyv8gkE8lIQzQLOE9qA4NdoeHch2"

# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
        color: #17365d;
    }

    h1, h2, h3, p {
        color: #17365d;
    }

    [data-testid="stWidgetLabel"] p {
        color: #17365d !important;
    }

    [data-testid="stSidebar"] {
        background-color: #202633;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: #ffffff !important;
    }

    .stTextInput input,
    .stNumberInput input {
        color: #ffffff !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        color: #ffffff;
    }

    div.stButton > button {
        width: 100%;
        background-color: #1769aa;
        color: white;
        border-radius: 8px;
        font-weight: 700;
        min-height: 45px;
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
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3. DOWNLOAD AND LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Download the trained model from Google Drive when it is
    not already present, then load the scikit-learn pipeline.
    """

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:

        # Remove an incomplete model file if present.
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        with st.spinner("Downloading trained car price model..."):
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
                "Model download failed. Check that the Google Drive "
                "file is shared with Anyone with the link as Viewer."
            )

    # The model was trained using scikit-learn 1.6.1.
    # Keep this dependency version consistent in requirements.txt.
    return joblib.load(MODEL_PATH)


# ============================================================
# 4. HEADER
# ============================================================

st.markdown(
    """
    <h1 style="text-align:center;">
        🚘 Indian Pre-Owned Car Price Prediction
    </h1>
    <p style="text-align:center;">
        Estimate the resale value of a used car using its
        specifications, age, and location.
    </p>
    """,
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

    st.subheader("Features considered")

    st.markdown(
        """
        - Manufacturer and model
        - Location and ownership
        - Manufacturing year and car age
        - Engine specifications
        - Fuel and transmission types
        - Body type and seating capacity
        - Vroom Audit Rating
        """
    )

    st.divider()

    st.caption(
        "Predictions are estimates, not guaranteed market offers."
    )

# ============================================================
# 6. INPUT FORM
# ============================================================

st.subheader("Enter Car Details")
st.write("Complete the fields below and click Predict Car Price.")

with st.form("car_prediction_form"):

    st.markdown("### 🚗 Basic Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        maker = st.selectbox(
            "Car Manufacturer",
            [
                "Maruti Suzuki",
                "Hyundai",
                "Honda",
                "Tata",
                "Mahindra",
                "Toyota",
                "Kia",
                "Renault",
                "Volkswagen",
                "Skoda",
                "Ford",
                "BMW",
                "Mercedes-Benz",
                "Audi",
                "Nissan",
                "MG",
                "Jeep",
                "Datsun",
                "Chevrolet",
                "Other"
            ]
        )

    with col2:
        model_name = st.text_input(
            "Car Model",
            value="Swift"
        ).strip()

    with col3:
        location = st.text_input(
            "City / Location",
            value="Delhi"
        ).strip()

    col1, col2, col3 = st.columns(3)

    with col1:
        manufacture_year = st.number_input(
            "Manufacturing Year",
            min_value=1980,
            max_value=2026,
            value=2020,
            step=1
        )

    with col2:
        age_of_car = st.number_input(
            "Age of Car (years)",
            min_value=0,
            max_value=60,
            value=6,
            step=1
        )

    with col3:
        distance = st.number_input(
            "Distance Driven (km)",
            min_value=0,
            max_value=2000000,
            value=35000,
            step=1000
        )

    st.markdown("### ⚙️ Engine and Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        engine_displacement = st.number_input(
            "Engine Displacement (cc)",
            min_value=0,
            max_value=10000,
            value=1197,
            step=1
        )

    with col2:
        engine_power = st.number_input(
            "Engine Power (bhp)",
            min_value=0.0,
            max_value=2000.0,
            value=82.0,
            step=1.0
        )

    with col3:
        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"]
        )

    st.markdown("### 🛞 Body and Transmission")

    col1, col2, col3 = st.columns(3)

    with col1:
        body_type = st.selectbox(
            "Body Type",
            [
                "Hatchback", "Sedan", "SUV", "MUV",
                "Coupe", "Convertible", "Minivan",
                "Pickup Truck", "Wagon", "Other"
            ]
        )

    with col2:
        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )

    with col3:
        owner_type = st.selectbox(
            "Owner Type",
            [
                "First", "Second", "Third", "Fourth & Above",
                "First Owner", "Second Owner", "Third Owner",
                "Fourth & Above Owner"
            ]
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        door_count = st.selectbox(
            "Number of Doors",
            [2, 3, 4, 5, 6],
            index=3
        )

    with col2:
        seat_count = st.selectbox(
            "Seating Capacity",
            [2, 4, 5, 6, 7, 8, 9, 10],
            index=2
        )

    with col3:
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
        type="primary",
        use_container_width=True
    )

# ============================================================
# 7. PREDICTION
# ============================================================

if submitted:

    if not model_name:
        st.error("Please enter the car model.")
        st.stop()

    if not location:
        st.error("Please enter the city or location.")
        st.stop()

    # Column names must match the cleaned training dataset.
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

    try:
        model = load_model()

        with st.spinner("Predicting the estimated car price..."):
            prediction = model.predict(input_data)
            predicted_price = float(prediction[0])

        if not pd.notna(predicted_price):
            st.error("The model returned an invalid prediction.")

        elif predicted_price < 0:
            st.warning(
                f"The model returned a negative prediction: "
                f"₹ {predicted_price:,.0f}. Please check the model."
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
                "This is an estimated price, not a guaranteed selling "
                "price. Actual value depends on vehicle condition, "
                "documentation, market demand, and negotiation."
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
            "Check the Google Drive download permissions, model file, "
            "feature names, and package compatibility."
        )

        st.code(f"{type(error).__name__}: {error}")
