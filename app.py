import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

# ---------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------
def create_model():
    model = Sequential([
        Input(shape=(8,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

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
    text-align: center;
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
st.markdown(
    "<h1>🩺 Breast Cancer Classification</h1>",
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD DATASET + MODEL
# ---------------------------------------------------
@st.cache_resource
def load_assets():

    df = pd.read_csv("data.csv")

    selected_features = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean"
    ]

    feature_frame = df[selected_features]

    scaler = MinMaxScaler()
    scaler.fit(feature_frame)

    model = create_model()
    model.load_weights("breast_cancer_weights.weights.h5")

    return scaler, model

scaler, model = load_assets()

# ---------------------------------------------------
# DESCRIPTION
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
# PREDICTION
# ---------------------------------------------------
if st.button("🔍 Predict Tumor Type"):

    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled, verbose=0)[0][0]

    st.header("🧠 Prediction Result")

    if prediction >= 0.5:
        st.error(
            f"⚠️ Malignant Tumor Detected\n\nConfidence: {prediction*100:.2f}%"
        )
    else:
        st.success(
            f"✅ Benign Tumor Detected\n\nConfidence: {(1-prediction)*100:.2f}%"
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