import streamlit as st
import numpy as np
import joblib

# Load Model
model = joblib.load("emotion_model.pkl")
emotions_map = {0: "Neutral 😐", 1: "Happy 😊", 2: "Sad 😢", 3: "Angry 😡"}

st.title("🎙️ Speech Emotion Recognition System")
st.write("CodeAlpha Machine Learning Internship - Task 2")

st.markdown("### Test Emotion Classification Model")

# Form for 5 key audio signal features
f1 = st.slider("MFCC Feature 1 (Pitch/Tone)", -3.0, 3.0, 0.0)
f2 = st.slider("MFCC Feature 2 (Energy)", -3.0, 3.0, 0.0)
f3 = st.slider("MFCC Feature 3 (Frequency)", -3.0, 3.0, 0.0)
f4 = st.slider("MFCC Feature 4 (Tempo)", -3.0, 3.0, 0.0)

if st.button("Classify Speech Emotion"):
    # Create feature vector matching model dimensions
    features = np.zeros((1, 40))
    features[0, :4] = [f1, f2, f3, f4]
    
    prediction = model.predict(features)[0]
    st.success(f"Detected Emotion: **{emotions_map[prediction]}**")
