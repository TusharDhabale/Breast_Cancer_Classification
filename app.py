import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background: linear-gradient(to bottom right, #0E1117, #1E1E2F);
    color: white;
}

h1, h2, h3 {
    color: #C084FC;
}

.stButton>button {
    background-color: #9333EA;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #7E22CE;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🩺 Breast Cancer Classification")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
@st.cache_resource
def load_assets():
    df = pd.read_csv("data.csv")
    feature_frame = df.drop(columns=["id", "diagnosis"], errors="ignore")
    feature_frame = feature_frame.loc[:, ~feature_frame.columns.str.contains("^Unnamed")]
    
    scaler = MinMaxScaler()
    scaler.fit(feature_frame)
    
    model = load_model("breast_cancer_model.keras", compile=False)
    return feature_frame.columns.tolist(), scaler, model, feature_frame

FEATURE_NAMES, scaler, model, feature_frame = load_assets()

# ---------------------------------------------------
# PROJECT DESCRIPTION
# ---------------------------------------------------
st.markdown("""
## 📌 Prediction Categories

- ✅ Benign (Non-Cancerous)
- ⚠️ Malignant (Cancerous)

Enter tumor feature values below.
""")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
st.header("📊 Tumor Feature Inputs")

col1, col2 = st.columns(2)

with col1:
    radius_mean = st.number_input("Radius Mean", value=14.0)
    texture_mean = st.number_input("Texture Mean", value=20.0)
    perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
    area_mean = st.number_input("Area Mean", value=600.0)

with col2:
    smoothness_mean = st.number_input("Smoothness Mean", value=0.10)
    compactness_mean = st.number_input("Compactness Mean", value=0.10)
    concavity_mean = st.number_input("Concavity Mean", value=0.10)
    concave_points_mean = st.number_input("Concave Points Mean", value=0.05)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
if st.button("🔍 Predict Tumor Type"):

    # Create a 30-feature input dict
    input_values = {name: 0.0 for name in FEATURE_NAMES}
    
    # Update with user inputs
    input_values[FEATURE_NAMES[0]] = radius_mean
    input_values[FEATURE_NAMES[1]] = texture_mean
    input_values[FEATURE_NAMES[2]] = perimeter_mean
    input_values[FEATURE_NAMES[3]] = area_mean
    input_values[FEATURE_NAMES[4]] = smoothness_mean
    input_values[FEATURE_NAMES[5]] = compactness_mean
    input_values[FEATURE_NAMES[6]] = concavity_mean
    input_values[FEATURE_NAMES[7]] = concave_points_mean
    
    # Create DataFrame with proper column names
    input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)
    
    # Scale input
    full_input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(full_input_scaled, verbose=0)

    confidence = prediction[0][0] * 100

    st.header("🧠 Prediction Result")

    if prediction[0][0] > 0.5:
        st.error(
            f"⚠️ Malignant Tumor Detected\n\nConfidence: {confidence:.2f}%"
        )
    else:
        st.success(
            f"✅ Benign Tumor Detected\n\nConfidence: {100-confidence:.2f}%"
        )

# ---------------------------------------------------
# DEMO INPUTS
# ---------------------------------------------------
st.markdown("---")

st.header("🧪 Demo Inputs")

demo1, demo2 = st.columns(2)

with demo1:
    st.success("""
    ✅ Benign Demo Input
    
    Radius Mean = 12.5  
    Texture Mean = 14.0  
    Perimeter Mean = 80  
    Area Mean = 450  
    Smoothness Mean = 0.09  
    Compactness Mean = 0.08  
    Concavity Mean = 0.05  
    Concave Points Mean = 0.03
    """)

with demo2:
    st.error("""
    ⚠️ Malignant Demo Input
    
    Radius Mean = 18.5  
    Texture Mean = 25.0  
    Perimeter Mean = 120  
    Area Mean = 1100  
    Smoothness Mean = 0.14  
    Compactness Mean = 0.20  
    Concavity Mean = 0.30  
    Concave Points Mean = 0.15
    """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown("""
<center>

### Minor Project Demonstration  
Department of Computer Science & Engineering  
JNCT Group of Colleges, Bhopal  

</center>
""", unsafe_allow_html=True)