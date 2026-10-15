
import streamlit as st
import pandas as pd
import joblib
import gdown
from pathlib import Path
from datetime import date

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AutoValue | Indian Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 2. DARK PREMIUM UI THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d18;
    --panel: #101827;
    --panel2: #141f31;
    --border: #25334a;
    --text: #f3f6fc;
    --muted: #99a9c0;
    --blue: #4f8cff;
    --blue2: #2864d7;
}

.stApp {
    background: radial-gradient(ellipse at top, #142440 0%, #080d18 48%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1240px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}

.hero {
    padding: 30px 32px;
    border: 1px solid #263956;
    border-radius: 22px;
    background: linear-gradient(120deg, #15243c 0%, #101827 58%, #101d34 100%);
    margin-bottom: 26px;
}

.eyebrow {
    color: #79aaff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(28px, 4vw, 44px);
    line-height: 1.15;
    letter-spacing: -1.4px;
    margin: 0 0 12px 0;
    font-weight: 800;
}

.hero p {
    color: #b2c1d7;
    font-size: 15px;
    line-height: 1.8;
    margin-bottom: 0;
    max-width: 690px;
}

.section-title {
    color: #f4f7fc;
    font-size: 19px;
    font-weight: 750;
    margin: 6px 0 4px 0;
}

.section-caption {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 17px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16, 24, 39, 0.88);
    border-color: var(--border) !important;
    border-radius: 18px !important;
}

div[data-testid="stWidgetLabel"] label {
    color: #d8e2f1 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

input, textarea {
    color: #f4f7fc !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
    background-color: #0b1321 !important;
    border-color: #2b3b54 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span {
    color: #f4f7fc !important;
}

.stNumberInput button {
    background: #17243a !important;
    color: white !important;
    border-color: #2b3b54 !important;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 11px;
    min-height: 46px;
    font-weight: 750;
    border: 1px solid #4f8cff;
    background: linear-gradient(100deg, #4f8cff, #2864d7);
    color: white;
    transition: all .2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border-color: #8db6ff;
    color: white;
    filter: brightness(1.12);
}

.price-card {
    background: linear-gradient(135deg, #142b4b, #101827 80%);
    border: 1px solid #31558a;
    border-radius: 20px;
    padding: 25px;
    margin-top: 12px;
}

.price-label {
    color: #a9c8ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}

.price-value {
    color: #ffffff;
    font-size: clamp(30px, 4vw, 44px);
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 12px 0;
    overflow-wrap: anywhere;
}

.price-note {
    color: #aebdd2;
    font-size: 12px;
    line-height: 1.6;
}

.metric-card {
    background: #101827;
    border: 1px solid #25334a;
    border-radius: 14px;
    padding: 16px;
    min-height: 85px;
}

.metric-label {
    color: #9aaac1;
    font-size: 12px;
    margin-bottom: 8px;
}

.metric-value {
    color: #f3f6fc;
    font-size: 17px;
    font-weight: 750;
    overflow-wrap: anywhere;
}

.footer {
    color: #7e8da4;
    text-align: center;
    font-size: 12px;
    padding-top: 26px;
}

.stAlert {
    border-radius: 12px;
}

hr {
    border-color: #25334a;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 1rem 2rem 1rem;
    }
    .hero {
        padding: 23px 20px;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. MODEL DOWNLOAD AND LOADING
#    Keep the existing Google Drive model and file ID.
# ============================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "car_price_model.joblib"

GOOGLE_DRIVE_FILE_ID = "1YDqqIyv8gkE8lIQzQLOE9qA4NdoeHch2"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        with st.spinner("Downloading the trained price prediction model..."):
            downloaded_file = gdown.download(
                id=GOOGLE_DRIVE_FILE_ID,
                output=str(MODEL_PATH),
                quiet=False,
            )

        if (
            downloaded_file is None
            or not MODEL_PATH.exists()
            or MODEL_PATH.stat().st_size == 0
        ):
            raise RuntimeError(
                "Model download failed. Check that the Google Drive file "
                "is shared with Anyone with the link as Viewer."
            )

    return joblib.load(MODEL_PATH)


# ============================================================
# 4. HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">AUTOVALUE · SMART VEHICLE VALUATION</div>
    <h1>Know what your car is worth.</h1>
    <p>
        Enter your vehicle specifications to estimate its pre-owned market
        price in India. Get a quick valuation powered by your trained
        machine-learning model.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. SIDEBAR INFORMATION
# ============================================================
with st.sidebar:
    st.markdown("## 🚘 AutoValue")
    st.caption("Indian Pre-Owned Car Price Prediction")
    st.divider()
    st.markdown("**How it works**")
    st.write("1. Enter your car details.")
    st.write("2. Submit the vehicle information.")
    st.write("3. View the model's estimated price.")
    st.divider()
    st.caption(
        "The estimate is generated by a machine-learning model. "
        "Actual selling prices may vary."
    )

# ============================================================
# 6. INPUT FORM
#    IMPORTANT: These feature names and their final order match
#    the existing trained model's expected input DataFrame.
# ============================================================
st.markdown('<div class="section-title">Vehicle configuration</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Enter the details as accurately as possible.</div>',
    unsafe_allow_html=True,
)

with st.form("vehicle_prediction_form"):
    # VEHICLE IDENTITY
    with st.container(border=True):
        st.markdown("### 🚗 Vehicle details")
        st.caption("Basic information about the car.")

        c1, c2, c3 = st.columns(3)

        with c1:
            maker = st.text_input(
                "Manufacturer / brand",
                value="Maruti",
                placeholder="e.g. Maruti, Hyundai, Honda",
                help="Enter the manufacturer name used in your dataset.",
            )

        with c2:
            model_name = st.text_input(
                "Car model",
                value="Swift",
                placeholder="e.g. Swift, City, Creta",
            )

        with c3:
            location = st.text_input(
                "City / location",
                value="Bangalore",
                placeholder="e.g. Bangalore, Delhi, Mumbai",
            )

        c4, c5, c6 = st.columns(3)

        with c4:
            current_year = date.today().year
            manufacture_year = st.number_input(
                "Manufacturing year",
                min_value=1980,
                max_value=current_year,
                value=min(2020, current_year),
                step=1,
            )

        with c5:
            age_of_car = st.number_input(
                "Age of car (years)",
                min_value=0,
                max_value=100,
                value=max(0, current_year - min(2020, current_year)),
                step=1,
                help="Enter the age expected by your training dataset.",
            )

        with c6:
            owner_type = st.selectbox(
                "Owner type",
                [
                    "First",
                    "Second",
                    "Third",
                    "Fourth & Above",
                    "First Owner",
                    "Second Owner",
                    "Third Owner",
                    "Fourth & Above Owner",
                ],
            )

    # USAGE AND POWERTRAIN
    with st.container(border=True):
        st.markdown("### ⚙️ Engine & performance")
        st.caption("Usage, engine specifications, and fuel information.")

        c1, c2, c3 = st.columns(3)

        with c1:
            distance = st.number_input(
                "Distance driven (km)",
                min_value=0,
                max_value=2_000_000,
                value=30000,
                step=1000,
            )

        with c2:
            engine_displacement = st.number_input(
                "Engine displacement (cc)",
                min_value=0,
                max_value=15000,
                value=1200,
                step=100,
            )

        with c3:
            engine_power = st.number_input(
                "Engine power (bhp)",
                min_value=0.0,
                max_value=2000.0,
                value=85.0,
                step=1.0,
            )

        c4, c5, c6 = st.columns(3)

        with c4:
            fuel_type = st.selectbox(
                "Fuel type",
                ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"],
            )

        with c5:
            transmission = st.selectbox(
                "Transmission",
                ["Manual", "Automatic"],
            )

        with c6:
            body_type = st.selectbox(
                "Body type",
                [
                    "Hatchback",
                    "Sedan",
                    "SUV",
                    "MUV",
                    "Coupe",
                    "Convertible",
                    "Minivan",
                    "Pickup Truck",
                    "Wagon",
                    "Other",
                ],
            )

    # VEHICLE STRUCTURE AND AUDIT
    with st.container(border=True):
        st.markdown("### 🛡️ Vehicle specifications")
        st.caption("Additional details used by the prediction model.")

        c1, c2, c3 = st.columns(3)

        with c1:
            door_count = st.number_input(
                "Number of doors",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
            )

        with c2:
            seat_count = st.number_input(
                "Number of seats",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
            )

        with c3:
            audit_rating = st.number_input(
                "Vroom Audit Rating",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Use the rating scale from your dataset.",
            )

    predict_clicked = st.form_submit_button(
        "✨  Predict estimated car price",
        use_container_width=True,
    )

# ============================================================
# 7. PREDICTION
#    Keep the exact 15 feature names and order used by the model.
# ============================================================
if predict_clicked:
    if not maker.strip() or not model_name.strip() or not location.strip():
        st.error("Please enter the manufacturer, car model, and location.")
    else:
        try:
            # Load the same trained pipeline from Google Drive.
            model_pipeline = load_model()

            # Do not rename or reorder these columns.
            input_data = pd.DataFrame(
                [[
                    maker.strip(),
                    model_name.strip(),
                    location.strip(),
                    distance,
                    owner_type,
                    manufacture_year,
                    age_of_car,
                    engine_displacement,
                    engine_power,
                    body_type,
                    audit_rating,
                    transmission,
                    door_count,
                    seat_count,
                    fuel_type,
                ]],
                columns=[
                    "Maker",
                    "model",
                    "Location",
                    "Distance",
                    "Owner Type",
                    "manufacture_year",
                    "Age of car",
                    "engine_displacement",
                    "engine_power",
                    "body_type",
                    "Vroom Audit Rating",
                    "transmission",
                    "door_count",
                    "seat_count",
                    "fuel_type",
                ],
            )

            with st.spinner("Analyzing vehicle details..."):
                prediction = model_pipeline.predict(input_data)
                predicted_price = float(prediction[0])

            if not pd.notna(predicted_price):
                st.error("The model returned an invalid prediction. Please try again.")
            elif predicted_price < 0:
                st.warning(
                    "The model returned a negative estimate. "
                    "Please verify the input details and try again."
                )
            else:
                st.markdown("---")
                st.markdown(
                    '<div class="section-title">Your valuation result</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-caption">'
                    'Estimated price based on the information provided.'
                    '</div>',
                    unsafe_allow_html=True,
                )

                result_left, result_right = st.columns([1.15, 1])

                with result_left:
                    st.markdown(
                        f"""
                        <div class="price-card">
                            <div class="price-label">Estimated car price</div>
                            <div class="price-value">₹ {predicted_price:,.0f}</div>
                            <div class="price-note">
                                Machine-learning estimate · Indian Rupees (INR)<br>
                                This is an estimate, not a guaranteed selling price.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with result_right:
                    st.markdown("#### Your vehicle")
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Manufacturer & model</div>
                            <div class="metric-value">
                                {maker.strip()} {model_name.strip()}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Manufacturing year · Age</div>
                            <div class="metric-value">
                                {int(manufacture_year)} · {int(age_of_car)} years
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.write("")
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Distance driven</div>
                            <div class="metric-value">{distance:,.0f} km</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("")
                with st.expander("View all submitted vehicle details"):
                    st.dataframe(
                        input_data.T.rename(columns={0: "Selected value"}),
                        use_container_width=True,
                    )

                st.caption(
                    "The estimate depends on the model's training data and "
                    "the selected inputs. Inspect the car and compare current "
                    "market listings before making a purchase or sale."
                )

        except Exception as error:
            st.error(
                "The prediction could not be completed. Check the model "
                "download and the app logs for details."
            )
            st.exception(error)

# ============================================================
# 8. FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    AUTOVALUE · INDIAN PRE-OWNED CAR PRICE PREDICTION<br>
    Built with Streamlit and a trained machine-learning pipeline
</div>
""", unsafe_allow_html=True)
