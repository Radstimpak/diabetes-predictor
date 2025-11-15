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

defaults = {
    "pregnancies": 1,
    "glucose": 100,
    "blood_pressure": 70,
    "skin_thickness": 20,
    "insulin": 80,
    "bmi": 32.0,
    "pedigree": 0.4,
    "age": 30
}

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    for key, value in defaults.items():
        st.session_state[key] = value

def reset_parameters():
    """Resets all input widgets to their default values."""
    for key, value in defaults.items():
        st.session_state[key] = value
    st.session_state.prediction_made = False

st.title('Pima Indians Diabetes Prediction')
st.write("Enter the patient's details below to predict the likelihood of diabetes.")

if model is not None and scaler is not None:
    
    st.subheader("Patient Input Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.number_input('Pregnancies', min_value=0, max_value=20, step=1, key='pregnancies')
        st.number_input('Glucose', min_value=0, max_value=200, step=1, key='glucose')
        st.number_input('BloodPressure', min_value=0, max_value=140, step=1, key='blood_pressure')
        st.number_input('SkinThickness', min_value=0, max_value=100, step=1, key='skin_thickness')
    
    with col2:
        st.number_input('Insulin', min_value=0, max_value=900, step=1, key='insulin')
        st.number_input('BMI', min_value=0.0, max_value=70.0, step=0.1, key='bmi')
        st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=3.0, step=0.01, key='pedigree')
        st.number_input('Age', min_value=0, max_value=120, step=1, key='age')

    st.write("---") 

    if st.button('Run Prediction'):
        st.session_state.prediction_made = True
        
        try:
            input_data = np.array([[
                st.session_state.pregnancies,
                st.session_state.glucose,
                st.session_state.blood_pressure,
                st.session_state.skin_thickness,
                st.session_state.insulin,
                st.session_state.bmi,
                st.session_state.pedigree,
                st.session_state.age
            ]])

            scaled_input_data = scaler.transform(input_data)

            prediction_prob = model.predict(scaled_input_data)[0][0]
            prediction_class = 1 if prediction_prob > 0.5 else 0

            st.subheader(f"Prediction Probability: {prediction_prob:.4f}")
            if prediction_class == 1:
                st.error("Result: **Positive for Diabetes** (Probability > 0.5)")
            else:
                st.success("Result: **Negative for Diabetes** (Probability <= 0.5)")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    if st.session_state.prediction_made:
        st.button('Reset Parameters', on_click=reset_parameters)

else:
    st.error("Model and/or scaler failed to load. Please check your files.")
