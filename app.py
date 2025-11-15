import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Pima Diabetes Predictor",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    """Load the saved Keras model and the scaler."""
    try:
        model = load_model('best_model.keras')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model_and_scaler()

def get_feature_names():
    return [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

st.title('🩺 Pima Indians Diabetes Prediction')
st.write("Enter the patient's details below to predict the likelihood of diabetes.")

if model is not None and scaler is not None:
    st.sidebar.header("Patient Input Features")
    st.sidebar.write("Adjust the sliders to match the patient's data.")

    feature_names = get_feature_names()
    user_inputs = {}
    
    user_inputs['Pregnancies'] = st.sidebar.number_input('Pregnancies', min_value=0, max_value=20, value=1, step=1)
    user_inputs['Glucose'] = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=100, step=1)
    user_inputs['BloodPressure'] = st.sidebar.number_input('BloodPressure', min_value=0, max_value=140, value=70, step=1)
    user_inputs['SkinThickness'] = st.sidebar.number_input('SkinThickness', min_value=0, max_value=100, value=20, step=1)
    user_inputs['Insulin'] = st.sidebar.number_input('Insulin', min_value=0, max_value=900, value=80, step=1)
    user_inputs['BMI'] = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    user_inputs['DiabetesPedigreeFunction'] = st.sidebar.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, value=0.4, step=0.01)
    user_inputs['Age'] = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30, step=1)

    st.subheader("Prediction")
    
    if st.button('Run Prediction'):
        try:
            input_data = np.array([[
                user_inputs['Pregnancies'],
                user_inputs['Glucose'],
                user_inputs['BloodPressure'],
                user_inputs['SkinThickness'],
                user_inputs['Insulin'],
                user_inputs['BMI'],
                user_inputs['DiabetesPedigreeFunction'],
                user_inputs['Age']
            ]])

            scaled_input_data =     .transform(input_data)

            prediction_prob = model.predict(scaled_input_data)[0][0]
            prediction_class = 1 if prediction_prob > 0.5 else 0

            st.subheader(f"Prediction Probability: {prediction_prob:.4f}")
            if prediction_class == 1:
                st.error("Result: **Positive for Diabetes** (Probability > 0.5)")
            else:
                st.success("Result: **Negative for Diabetes** (Probability <= 0.5)")
                
            with st.expander("Show Scaled Input Data (for debugging)"):
                st.write(scaled_input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.error("Model and/or scaler failed to load. Please check your files.")
