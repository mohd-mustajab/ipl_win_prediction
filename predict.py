import streamlit as st
import joblib
import numpy as np
import os

# Ensure all files are in the same directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(filename):
    return joblib.load(os.path.join(BASE_DIR, filename))

# Load models/encoders
model = load_model("model.pkl")
team_encoder = load_model("team_encoder.pkl")
venue_encoder = load_model("venue_encoder.pkl")
toss_decision_encoder = load_model("toss_decision_encoder.pkl")

st.title("🏏 IPL Match Winner Predictor")

team_names = list(team_encoder.classes_)
venue_names = list(venue_encoder.classes_)
toss_decision_options = list(toss_decision_encoder.classes_)

# UI
team1 = st.selectbox("Team 1", team_names)
team2 = st.selectbox("Team 2", [t for t in team_names if t != team1])
venue = st.selectbox("Venue", venue_names)
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", toss_decision_options)

if st.button("Predict Winner"):
    try:
        input_data = np.array([
            team_encoder.transform([team1])[0],
            team_encoder.transform([team2])[0],
            venue_encoder.transform([venue])[0],
            team_encoder.transform([toss_winner])[0],
            toss_decision_encoder.transform([toss_decision])[0],
        ]).reshape(1, -1)

        predicted_proba = model.predict_proba(input_data)[0]
        predicted_class = model.predict(input_data)[0]
        predicted_team = team_encoder.inverse_transform([predicted_class])[0]
        confidence = predicted_proba[predicted_class] * 100

        st.success(f"🏆 Predicted Winner: {predicted_team}")
        st.info(f"📊 Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

