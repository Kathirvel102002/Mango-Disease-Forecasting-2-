import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Mango Disease Forecast System",
    page_icon="🥭",
    layout="wide"
)

# ==========================================================
# SIDEBAR MENU
# ==========================================================

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "🦠 Disease Description",
        "📊 Disease Prediction",
        "👨‍💻 About Us"
    ]
)

# ==========================================================
# HOME PAGE
# ==========================================================

if menu == "🏠 Home":

    st.title("🥭 Mango Disease Forecast System")

    st.image(
        "https://images.unsplash.com/photo-1591073113125-e46713c829ed",
        use_container_width=True
    )

    st.markdown("""
    ## Welcome

    Mango is one of the most important fruit crops cultivated in Theni District
    and many other regions of Tamil Nadu.

    Disease outbreaks can significantly reduce fruit yield and quality.

    This application uses machine learning and weather parameters to forecast
    major mango diseases one week in advance and support timely disease
    management decisions.

    ### Objectives

    ✅ Early disease forecasting

    ✅ Weather-based prediction

    ✅ Disease risk assessment

    ✅ Support for farmers and extension workers

    ### Diseases Covered

    - Leaf Anthracnose
    - Black Banded
    - Red Rust
    - Die Back
    - Sooty Mould
    """)

# ==========================================================
# DISEASE DESCRIPTION PAGE
# ==========================================================

elif menu == "🦠 Disease Description":

    st.title("🦠 Mango Disease Description")

    disease_info = st.selectbox(
        "Select Disease",
        [
            "Leaf Anthracnose",
            "Black Banded",
            "Red Rust",
            "Die Back",
            "Sooty Mould"
        ]
    )

    if disease_info == "Leaf Anthracnose":

        st.subheader("Leaf Anthracnose")

        st.write("""
        **Causal Organism:** Colletotrichum gloeosporioides

        **Symptoms**
        - Small dark brown lesions on leaves
        - Premature leaf fall
        - Fruit spotting and rotting

        **Favorable Conditions**
        - High humidity
        - Frequent rainfall
        - Warm temperature
        """)

    elif disease_info == "Black Banded":

        st.subheader("Black Banded")

        st.write("""
        **Symptoms**
        - Black band-like lesions on branches
        - Reduced vigour
        - Twig drying

        **Favorable Conditions**
        - Wet weather
        - High humidity
        """)

    elif disease_info == "Red Rust":

        st.subheader("Red Rust")

        st.write("""
        **Causal Organism:** Cephaleuros virescens

        **Symptoms**
        - Orange-red velvety spots on leaves
        - Reduced photosynthesis
        - Leaf yellowing

        **Favorable Conditions**
        - Humid environment
        - Warm weather
        """)

    elif disease_info == "Die Back":

        st.subheader("Die Back")

        st.write("""
        **Symptoms**
        - Drying from shoot tip downward
        - Branch death
        - Gum exudation

        **Favorable Conditions**
        - High humidity
        - Poor orchard sanitation
        """)

    elif disease_info == "Sooty Mould":

        st.subheader("Sooty Mould")

        st.write("""
        **Symptoms**
        - Black fungal growth on leaves
        - Reduced photosynthesis
        - Poor fruit quality

        **Favorable Conditions**
        - Presence of honeydew insects
        - Humid weather
        """)

# ==========================================================
# DISEASE PREDICTION PAGE
# ==========================================================
# ==========================================================
# PAGE CONFIG
# ==========================================================

# ==========================================================
# DISEASE PREDICTION PAGE
# ==========================================================

elif menu == "📊 Disease Prediction":

    st.title("🥭 Mango Disease Forecast System")
    st.markdown("### Forecast Next Week Disease Severity")

   # ==========================================================
# INPUTS
# ==========================================================

disease = st.selectbox(
    "Select Disease",
    [
        "Leaf Anthracnose",
        "Black Banded",
        "Red Rust",
        "Die Back",
        "Sooty Mould"
    ]
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Weather Parameters")

    rf = st.number_input(
        "Rainfall (RF) mm",
        min_value=0.0,
        value=25.0
    )

    rd = st.number_input(
        "Rainy Days (RD)",
        min_value=0,
        value=2
    )

    rh = st.number_input(
        "Humidity (RH) %",
        min_value=0.0,
        max_value=100.0,
        value=80.0
    )

    tmax = st.number_input(
        "Maximum Temperature (°C)",
        value=34.0
    )

    tmin = st.number_input(
        "Minimum Temperature (°C)",
        value=25.0
    )

with col2:
    st.subheader("Field Parameters")

    current_disease = st.number_input(
        "Current Disease Severity (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0
    )

    week = st.number_input(
        "Week Number (1-52)",
        min_value=1,
        max_value=52,
        value=24
    )

# ==========================================================
# MODEL FILES
# ==========================================================

model_files = {
    "Leaf Anthracnose": "LEAF ANTHRACNOSE_farmer_forecast.pkl",
    "Black Banded": "BLACK_BANDED_farmer_forecast.pkl",
    "Red Rust": "RED RUST_farmer_forecast.pkl",
    "Die Back": "DIE BACK_farmer_forecast.pkl",
    "Sooty Mould": "SOOTY MOULD_farmer_forecast.pkl"
}

# ==========================================================
# PREDICTION
# ==========================================================

if st.button("Forecast Disease Risk"):

    try:

        T_avg = (tmax + tmin) / 2
        T_Range = tmax - tmin

        week_sin = np.sin(2 * np.pi * week / 52)
        week_cos = np.cos(2 * np.pi * week / 52)

        input_data = pd.DataFrame([[
            rf,
            rd,
            rh,
            tmax,
            tmin,
            T_avg,
            T_Range,
            current_disease,
            week_sin,
            week_cos
        ]], columns=[
            "RF",
            "RD",
            "RH",
            "T_MAX",
            "T_MIN",
            "T_avg",
            "T_Range",
            "DISEASE",
            "week_sin",
            "week_cos"
        ])

        model_file = model_files[disease]

        if not os.path.exists(model_file):
            st.error(f"Model file not found: {model_file}")
            st.stop()

        model = joblib.load(model_file)

        forecast = float(model.predict(input_data)[0])

        if forecast < 20:
            risk = "🟢 Low"
            advice = "Routine monitoring is sufficient."

        elif forecast < 40:
            risk = "🟡 Moderate"
            advice = "Inspect orchard regularly."

        elif forecast < 60:
            risk = "🟠 High"
            advice = "Start disease management measures."

        else:
            risk = "🔴 Epidemic"
            advice = "Immediate control measures required."

        st.success("Forecast Completed")

        st.metric(
            label="Predicted Disease Severity (%)",
            value=f"{forecast:.2f}"
        )

        st.subheader("Risk Assessment")
        st.write(f"**Risk Level:** {risk}")
        st.write(f"**Recommendation:** {advice}")

    except Exception as e:
        st.error(str(e))
# ==========================================================
# ABOUT US PAGE
# ==========================================================

elif menu == "👨‍💻 About Us":

    st.title("👨‍💻 About Us")

    st.markdown("""
    ## Mango Disease Forecast System

    This application was developed for forecasting major mango diseases
    using weather parameters and machine learning techniques.

    ### Developer

    **Mr. K. Kathirvel M.**

    PG Student

    Department of Horticulture

    Agricultural College and Research Institute

    Tamil Nadu Agricultural University

    ### Study Area

    Theni District, Tamil Nadu, India

    ### Technologies Used

    - Python
    - Streamlit
    - Machine Learning
    - Pandas
    - NumPy
    - Joblib

    ### Purpose

    To provide an early warning system for mango disease outbreaks and
    improve disease management decisions.
    """)