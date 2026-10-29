import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained predictive assets we downloaded from Google Colab
try:
    model = joblib.load('student_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    model = None
    model_columns = None

st.set_page_config(page_title="Student Engagement Dashboard", layout="wide")

st.title("🎓 Student Participation Predictive Dashboard")
st.write("Input prospective student attributes below to calculate operational sign-up probabilities.")

# Dashboard Interactive Sidebar Controls
st.sidebar.header("🎯 Student & Opportunity Attributes")
gpa = st.sidebar.slider("Current Student GPA", 0.0, 4.0, 3.2, 0.1)
past_attendance = st.sidebar.slider("Past Attendance Rate (%)", 0, 100, 75) / 100.0

category = st.sidebar.selectbox("Opportunity Category", ["workshop", "webinar", "career fair", "networking"])
gender = st.sidebar.selectbox("Gender", ["male", "female", "other"])
country = st.sidebar.text_input("Country Name", "united states").lower().strip()
major = st.sidebar.text_input("Current/Intended Major", "computer science").lower().strip()

# Run Prediction Logic
if st.button("📊 Calculate Success Probability"):
    if model is None:
        st.error("Error: 'student_model.pkl' asset file missing in directory path.")
    else:
        # Construct template base match array initialized to zero
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0
        
        # Inject continuous scalar variables if they were used during encoding
        if 'Signup_Year_Num' in input_df.columns: input_df.loc[0, 'Signup_Year_Num'] = 2026
        if 'Signup_Month_Num' in input_df.columns: input_df.loc[0, 'Signup_Month_Num'] = 10
        
        # Map selected categorical features to their respective dummy text flag keys
        cat_keys = [
            f"opportunity category_{category}",
            f"gender_{gender}",
            f"country_{country}",
            f"current/intended major_{major}"
        ]
        
        for key in cat_keys:
            if key in input_df.columns:
                input_df.loc[0, key] = 1
                
        # Extract percentage odds from our random forest brain
        probability = model.predict_proba(input_df)[0, 1] * 100
        
        # Display Outcome Meter
        st.metric(label="Calculated Likelihood of Successful Student Sign-Up", value=f"{probability:.2f}%")
        
        if probability >= 75:
            st.success("🟢 High Engagement Cluster: Highly recommended opportunity pairing.")
        elif probability >= 40:
            st.warning("🟡 Moderate Risk Profile: Outreaching support recommended.")
        else:
            st.error("🔴 High Risk Drop-Off Zone: Structural interventions required.")