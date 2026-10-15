
import streamlit as st
import pandas as pd
import joblib
import gdown
import html
from pathlib import Path
from datetime import date

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AutoValue | Luxury Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. LUXURY BLACK & GOLD THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #05080c;
    --panel: #0b1016;
    --panel2: #101720;
    --gold: #e8b95b;
    --gold-light: #ffdf91;
    --border: #45371f;
    --text: #f6f3eb;
    --muted: #a5acb8;
}

.stApp {
    background:
        radial-gradient(ellipse at top right, #211a0d 0%, transparent 36%),
        linear-gradient(145deg, #070a0e, #05080c 65%);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3, p, label, div {
    font-family: 'Inter', sans-serif;
}

.brand {
    color: var(--gold);
    font-weight: 800;
    font-size: 23px;
    letter-spacing: -0.7px;
}

.brand-sub {
    color: #b7bec8;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.hero {
    border: 1px solid #55401e;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 24px;
    background:
        radial-gradient(ellipse at 95% 20%, #4a3415 0%, transparent 32%),
        linear-gradient(110deg, #11151a, #0b0f13 65%, #17120b);
}

.eyebrow {
    color: var(--gold);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin-bottom: 12px;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(29px, 4vw, 46px);
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0 0 12px 0;
}

.hero h1 span {
    color: var(--gold);
}

.hero p {
    color: #c1c5cb;
    font-size: 14px;
    line-height: 1.8;
    max-width: 740px;
}

.section-heading {
    color: var(--gold-light);
    font-size: 19px;
    font-weight: 750;
    margin: 4px 0;
}

.section-caption {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 16px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(11, 16, 22, 0.94);
    border: 1px solid var(--border) !important;
    border-radius: 15px !important;
}

div[data-testid="stWidgetLabel"] label {
    color: #e5dfd2 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
    background-color: #0c121a !important;
    border-color: #383a3a !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    color: #f5f1e8 !important;
}

input {
    color: #f5f1e8 !important;
}

.stNumberInput button {
    background: #19170f !important;
    color: var(--gold-light) !important;
    border-color: #45371f !important;
}

.stButton > button,
.stFormSubmitButton > button {
    min-height: 45px;
    border-radius: 9px;
    font-weight: 750;
    background: linear-gradient(105deg, #f1cc79, #c28b32);
    color: #15100a;
    border: 1px solid #ffdf91;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(105deg, #ffe6a7, #dba94e);
    color: #100d08;
    border-color: #ffe6a7;
}

.price-card {
    border: 1px solid #9a702e;
    border-radius: 15px;
    padding: 24px;
    background:
        radial-gradient(ellipse at top left, #3d2d13, transparent 65%),
        linear-gradient(145deg, #15130e, #0b1016);
    margin: 10px 0 18px 0;
}

.price-label {
    color: var(--gold-light);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.price-value {
    color: var(--gold-light);
    font-size: clamp(29px, 3.5vw, 42px);
    font-weight: 800;
    letter-spacing: -1px;
    margin: 12px 0;
    overflow-wrap: anywhere;
}

.price-note {
    color: #b6bbc2;
    font-size: 11px;
    line-height: 1.7;
}

.summary-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 2px;
    border-bottom: 1px solid #242a30;
    font-size: 12px;
}

.summary-label {
    color: #aab2bc;
}

.summary-value {
    color: #f5f1e8;
    font-weight: 600;
    text-align: right;
    overflow-wrap: anywhere;
}

.gold-rule {
    border: 0;
    height: 1px;
    background: linear-gradient(90deg, #a77a30, #302719, transparent);
    margin: 20px 0;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f13, #07090c);
    border-right: 1px solid #382c1b;
}

div[data-testid="stSidebar"] * {
    color: #e9e3d7;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

.footer {
    color: #888b91;
    font-size: 11px;
    text-align: center;
    padding: 22px 0 8px 0;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1rem 0.8rem;
    }
    .hero {
        padding: 21px;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. MODEL CONFIGURATION
# Keep the existing Google Drive file ID.
# ============================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "car_price_model.joblib"
GOOGLE_DRIVE_FILE_ID = "1YDqqIyv8gkE8lIQzQLOE9qA4NdoeHch2"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

        with st.spinner("Preparing the trained valuation model..."):
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
                "Model download failed. Confirm that the Google Drive "
                "file is shared as Anyone with the link — Viewer."
            )

    return joblib.load(MODEL_PATH)


# ============================================================
# 4. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="brand">🚘 AutoValue</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">Indian pre-owned car valuation</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    st.markdown("### About AutoValue")
    st.write(
        "An ML-powered interface for estimating the price of "
        "pre-owned vehicles in India."
    )

    st.markdown("### Features considered")
    for feature in [
        "Manufacturer and model",
        "City and ownership",
        "Manufacturing year and age",
        "Distance driven",
        "Engine specifications",
        "Fuel and transmission",
        "Body type and seating",
        "Vroom Audit Rating",
    ]:
        st.markdown(f"✓ {feature}")

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
    st.caption(
        "Predictions are estimates and are not guaranteed market offers."
    )

# ============================================================
# 5. LUXURY HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ PREMIUM VEHICLE VALUATION</div>
    <h1>Know what your car <span>is worth.</span></h1>
    <p>
        A refined, data-driven way to estimate your vehicle's value.
        Enter the car specifications below to receive a machine-learning
        estimate in Indian Rupees.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 6. VEHICLE INPUT FORM
# IMPORTANT:
# The feature names and order below preserve the existing
# model input structure. Do not change them without checking
# the model's training notebook.
# ============================================================
st.markdown(
    '<div class="section-heading">Vehicle configuration</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-caption">'
    'Complete the three sections below to estimate your car value.'
    '</div>',
    unsafe_allow_html=True,
)

with st.form("luxury_vehicle_form"):
    # VEHICLE DETAILS
    with st.container(border=True):
        st.markdown("#### 🚘  1. Vehicle details")
        st.caption("Basic information and ownership.")

        c1, c2, c3 = st.columns(3)

        with c1:
            maker = st.text_input(
                "Manufacturer / Brand",
                value="Maruti",
                placeholder="e.g. Maruti",
            )

        with c2:
            model_name = st.text_input(
                "Car model",
                value="Swift",
                placeholder="e.g. Swift",
            )

        with c3:
            location = st.text_input(
                "City / Location",
                value="Bangalore",
                placeholder="e.g. Bangalore",
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            current_year = date.today().year
            manufacture_year = st.number_input(
                "Manufacturing year",
                min_value=1980,
                max_value=current_year,
                value=min(2020, current_year),
                step=1,
            )

        with c2:
            age_of_car = st.number_input(
                "Age of car (years)",
                min_value=0,
                max_value=100,
                value=max(0, current_year - min(2020, current_year)),
                step=1,
            )

        with c3:
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

    # ENGINE AND PERFORMANCE
    with st.container(border=True):
        st.markdown("#### ⚙️  2. Engine & performance")
        st.caption("Usage, powertrain, and vehicle classification.")

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

        c1, c2, c3 = st.columns(3)

        with c1:
            fuel_type = st.selectbox(
                "Fuel type",
                ["Petrol", "Diesel", "CNG", "LPG", "Electric", "Hybrid"],
            )

        with c2:
            transmission = st.selectbox(
                "Transmission",
                ["Manual", "Automatic"],
            )

        with c3:
            body_type = st.selectbox(
                "Body type",
                [
                    "Hatchback", "Sedan", "SUV", "MUV", "Coupe",
                    "Convertible", "Minivan", "Pickup Truck", "Wagon",
                    "Other",
                ],
            )

    # VEHICLE SPECIFICATIONS
    with st.container(border=True):
        st.markdown("#### 🛡️  3. Vehicle specifications")
        st.caption("Additional attributes used by the model.")

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
            )

    predict_clicked = st.form_submit_button(
        "✦  PREDICT ESTIMATED CAR PRICE",
        use_container_width=True,
    )

# ============================================================
# 7. PREDICTION — SAME TRAINED PIPELINE
# ============================================================
if predict_clicked:
    if not maker.strip() or not model_name.strip() or not location.strip():
        st.error("Please fill in the manufacturer, model, and city.")
    else:
        try:
            model_pipeline = load_model()

            # Keep these exact 15 feature names and their order.
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

            with st.spinner("Calculating your vehicle valuation..."):
                prediction = model_pipeline.predict(input_data)
                predicted_price = float(prediction[0])

            if not pd.notna(predicted_price):
                st.error("The model returned an invalid price.")
            elif predicted_price < 0:
                st.warning(
                    "The model returned a negative estimate. "
                    "Check the inputs and your training data."
                )
            else:
                safe_maker = html.escape(maker.strip())
                safe_model = html.escape(model_name.strip())
                safe_location = html.escape(location.strip())
                safe_fuel = html.escape(fuel_type)
                safe_transmission = html.escape(transmission)
                safe_body = html.escape(body_type)

                st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-heading">✦ Your valuation</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-caption">'
                    'Your estimated value based on the submitted details.'
                    '</div>',
                    unsafe_allow_html=True,
                )

                result_col, summary_col = st.columns([1.05, 1])

                with result_col:
                    st.markdown(
                        f"""
                        <div class="price-card">
                            <div class="price-label">✦ Estimated car price</div>
                            <div class="price-value">₹ {predicted_price:,.0f}</div>
                            <div class="price-note">
                                MACHINE-LEARNING ESTIMATE · INR<br><br>
                                This is an estimate, not a guaranteed selling price.
                                Actual market value can vary with condition,
                                location, and current demand.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with summary_col:
                    with st.container(border=True):
                        st.markdown("#### Vehicle summary")

                        summary_items = [
                            ("Manufacturer & model",
                             f"{safe_maker} {safe_model}"),
                            ("Location", safe_location),
                            ("Manufacturing year", str(int(manufacture_year))),
                            ("Age of car", f"{int(age_of_car)} years"),
                            ("Distance driven", f"{distance:,.0f} km"),
                            ("Fuel type", safe_fuel),
                            ("Transmission", safe_transmission),
                            ("Body type", safe_body),
                            ("Seating capacity", str(int(seat_count))),
                            ("Audit rating", f"{audit_rating:.1f} / 10"),
                        ]

                        for label, value in summary_items:
                            st.markdown(
                                f"""
                                <div class="summary-row">
                                    <span class="summary-label">{label}</span>
                                    <span class="summary-value">{value}</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                with st.expander("View all submitted model inputs"):
                    st.dataframe(
                        input_data.T.rename(columns={0: "Selected value"}),
                        use_container_width=True,
                    )

                st.caption(
                    "Before buying or selling, compare this estimate with "
                    "current listings and the vehicle's inspected condition."
                )

        except Exception as error:
            st.error(
                "Prediction failed. Check the app logs and confirm that "
                "the model file and input columns are correct."
            )
            st.exception(error)

# ============================================================
# 8. FOOTER
# ============================================================
st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <strong style="color:#e8b95b;">AUTOVALUE</strong>
    &nbsp; · &nbsp; INDIAN PRE-OWNED CAR PRICE PREDICTION<br>
    Powered by Streamlit and your trained machine-learning pipeline
</div>
""", unsafe_allow_html=True)
