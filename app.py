# app.py - Interactive Iris Flower Classification Web App
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

# Title and description
st.title("🌸 Iris Flower Species Classifier")
st.markdown("""
This app predicts the species of an Iris flower based on its measurements.
Enter the values below and click **Predict**.
""")

# Load the trained model and scaler (we'll use the best one you saved)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_RandomForest.pkl')  # Change if your best model has different name
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except:
        st.error("Model not found! Please run `python main.py` first to train and save the model.")
        return None, None

model, scaler = load_model()

if model is None:
    st.stop()

# Input sliders (user-friendly)
st.subheader("Enter Flower Measurements")

sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.0, step=0.1)
sepal_width  = st.slider("Sepal Width (cm)",  min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width  = st.slider("Petal Width (cm)",  min_value=0.1, max_value=2.5, value=1.0, step=0.1)

# Prediction button
if st.button("🔮 Predict Species"):
    # Prepare input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display result with nice colors
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"**Predicted Species: {prediction}**")
    
    # Show confidence
    st.subheader("Prediction Confidence")
    for i, sp in enumerate(species):
        st.progress(probability[i], text=f"{sp}: {probability[i]:.1%}")

    # Optional: Show sample image (you can add later)
    st.info("Tip: Setosa usually has small petals, Virginica has larger ones.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info("""
This app uses a trained **Random Forest** model on the classic Iris dataset.
It demonstrates how machine learning models can be deployed as interactive web apps.
""")

st.sidebar.markdown("Made with ❤️ using Streamlit + scikit-learn")
