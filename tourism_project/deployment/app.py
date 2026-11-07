import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing the Wellness Tourism Package
based on their profile and sales interaction details.
Enter the customer information below to get a prediction.
""")

# ---------------- Customer Details ----------------
st.header("Customer Details")
age             = st.number_input("Age", 18, 100, 35)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier       = st.selectbox("City Tier", [1, 2, 3])
occupation      = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender          = st.selectbox("Gender", ["Male", "Female"])
num_persons     = st.number_input("Number of Persons Visiting", 1, 10, 2)
property_star   = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips       = st.number_input("Number of Trips (per year)", 0, 30, 3)
passport        = st.selectbox("Has Passport?", ["Yes", "No"])
own_car         = st.selectbox("Owns a Car?", ["Yes", "No"])
num_children    = st.number_input("Number of Children Visiting (under 5)", 0, 5, 0)
designation     = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income  = st.number_input("Monthly Income", 0.0, 1_000_000.0, 25_000.0, 500.0)

# ---------------- Interaction Data ----------------
st.header("Sales Interaction")
pitch_score     = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
num_followups   = st.number_input("Number of Follow-ups", 0, 10, 3)
pitch_duration  = st.number_input("Duration of Pitch (min)", 0.0, 200.0, 15.0, 1.0)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": num_followups,
    "DurationOfPitch": pitch_duration,
}])

# ---------------- Predict ----------------
if st.button("Predict Purchase"):
    # Use the same probability threshold the model was evaluated with
    probability = model.predict_proba(input_data)[0][1]
    will_buy = probability >= 0.50

    st.subheader("Prediction Result")
    if will_buy:
        st.success(f"Likely to purchase (probability {probability:.1%})")
    else:
        st.warning(f"Unlikely to purchase (probability {probability:.1%})")
    st.caption(f"Decision threshold: {0.50:.1%}")

    with st.expander("Inputs sent to the model"):
        st.dataframe(input_data)
