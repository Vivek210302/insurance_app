import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_lottie import st_lottie
import json

# ----------------------------------
# Load Model
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load('optimized_random_forest_model.joblib')

model = load_model()

# ----------------------------------
# Page Config + Global CSS
# ----------------------------------
st.set_page_config(page_title="Insurance Charge Predictor", layout="wide", page_icon="💼")

st.markdown(
    """
    <style>
        .title { font-size: 45px; text-align: center; font-weight: 700; color: #2E86C1; }
        .subtitle { font-size: 20px; text-align: center; color: #444; margin-bottom: 25px; }
        .result-card {
            padding: 25px;
            background: linear-gradient(135deg, #D4EFDF, #ABEBC6);
            border-radius: 15px;
            text-align: center;
            border: 1px solid #A9DFBF;
        }
        .footer { font-size: 14px; color: #999; text-align: center; margin-top: 50px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
menu = st.sidebar.radio("Navigation", ["🏠 Home", "📈 Predict", "📊 Analytics", "📁 Upload Dataset"])

# ----------------------------------
# Lottie Animation Loader
# ----------------------------------
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)


# Load animation if available
try:
    animation = load_lottie("animation.json")
except:
    animation = None

# ----------------------------------
# HOME PAGE
# ----------------------------------
if menu == "🏠 Home":

    st.markdown('<div class="title">💼 Insurance Cost Prediction App</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A modern ML-powered tool to estimate medical insurance charges</div>', unsafe_allow_html=True)

    if animation:
        st_lottie(animation, height=300)
    else:
        st.info("Animation file not found. Add animation.json for animated graphics.")

    st.write("---")
    st.subheader("✨ Features")
    st.write("""
    - Intelligent ML model prediction
    - Clean 3‑column professional UI
    - Interactive analytics (Plotly charts)
    - Upload your own dataset
    - Side‑by‑side navigation for smooth experience
    """)


# ----------------------------------
# PREDICT PAGE
# ----------------------------------
if menu == "📈 Predict":

    st.markdown('<div class="title">📈 Insurance Charge Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Fill the details below</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 0, 100)
        bmi = st.number_input("BMI", 0.0, 80.0)

    with col2:
        children = st.number_input("Children", 0, 10)
        smoker = st.selectbox("Smoker", ["yes", "no"])

    with col3:
        sex = st.selectbox("Sex", ["male", "female"])
        region = st.selectbox("Region", ["northwest", "southeast", "southwest"])

    # One‑hot encoding
    input_df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [1 if smoker == "yes" else 0],
        'sex_male': [1 if sex == "male" else 0],
        'region_northwest': [1 if region == "northwest" else 0],
        'region_southeast': [1 if region == "southeast" else 0],
        'region_southwest': [1 if region == "southwest" else 0]
    })

    if st.button("Predict", use_container_width=True):
        prediction = model.predict(input_df)[0]
        st.markdown(
            f"""
            <div class='result-card'>
                <h3>Predicted Charge</h3>
                <h1><b>${prediction:.2f}</b></h1>
            </div>
            """,
            unsafe_allow_html=True
        )


# ----------------------------------
# ANALYTICS PAGE
# ----------------------------------
if menu == "📊 Analytics":

    st.markdown('<div class="title">📊 Data Analytics</div>', unsafe_allow_html=True)

    try:
        df = pd.read_csv("insurance.csv")

        tab1, tab2, tab3 = st.tabs(["Age vs Charges", "BMI vs Charges", "Smoker Impact"])

        # Age vs Charges
        with tab1:
            fig = px.scatter(df, x="age", y="charges", color="sex", title="Age vs Charges")
            st.plotly_chart(fig, use_container_width=True)

        # BMI vs Charges
        with tab2:
            fig = px.scatter(df, x="bmi", y="charges", color="smoker", title="BMI vs Charges")
            st.plotly_chart(fig, use_container_width=True)

        # Smoker Impact
        with tab3:
            fig = px.box(df, x="smoker", y="charges", title="Smoker vs Non‑Smoker Charges")
            st.plotly_chart(fig, use_container_width=True)

    except:
        st.error("Dataset 'insurance.csv' not found. Upload it in the next tab.")

# ----------------------------------
# UPLOAD DATASET
# ----------------------------------
if menu == "📁 Upload Dataset":
    st.markdown('<div class="title">📁 Upload Your Dataset</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### Preview")
        st.dataframe(df)
        st.success("Dataset loaded successfully!")

# ----------------------------------
# FOOTER
# ----------------------------------
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)
