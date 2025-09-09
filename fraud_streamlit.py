import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('insurance_fraud.sav', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'insurance_fraud_model.sav' not found. Please ensure the file is in the same directory.")
        return None

# Main app
def main():
    st.title("🚗 Insurance Fraud Detection System")
    st.markdown("Enter the insurance claim details below to predict potential fraud.")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Claim Information")
        
        # Month
        month = st.selectbox(
            "Month",
            ["Dec", "Jan", "Oct", "Jun", "Feb", "Nov", "Apr", "Mar", "Aug", "Jul", "May", "Sep"]
        )
        
        # Day of Week
        day_of_week = st.selectbox(
            "Day of Week",
            ["Wednesday", "Friday", "Saturday", "Monday", "Tuesday", "Sunday", "Thursday"]
        )
        
        # Vehicle Make
        make = st.selectbox(
            "Vehicle Make",
            ["Honda", "Toyota", "Ford", "Mazda", "Chevrolet", "Pontiac", "Accura", "Dodge", 
             "Mercury", "Jaguar", "Nissan", "VW", "Saab", "Saturn", "Porche", "BMW", 
             "Mercedes", "Ferrari", "Lexus"]
        )
        
        # Accident Area
        accident_area = st.selectbox(
            "Accident Area",
            ["Urban", "Rural"]
        )
        
        # Day of Week Claimed
        day_of_week_claimed = st.selectbox(
            "Day of Week Claimed",
            ["Tuesday", "Monday", "Thursday", "Friday", "Wednesday", "Saturday", "Sunday", "0"]
        )
        
        # Month Claimed
        month_claimed = st.selectbox(
            "Month Claimed",
            ["Jan", "Nov", "Jul", "Feb", "Mar", "Dec", "Apr", "Aug", "May", "Jun", "Sep", "Oct", "0"]
        )
        
        # Sex
        sex = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )
        
        # Marital Status
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Widow", "Divorced"]
        )
        
        # Fault
        fault = st.selectbox(
            "Fault",
            ["Policy Holder", "Third Party"]
        )
        
        # Policy Type
        policy_type = st.selectbox(
            "Policy Type",
            ["Sport - Liability", "Sport - Collision", "Sedan - Liability", "Utility - All Perils",
             "Sedan - All Perils", "Sedan - Collision", "Utility - Collision", "Utility - Liability",
             "Sport - All Perils"]
        )
        
        # Vehicle Category
        vehicle_category = st.selectbox(
            "Vehicle Category",
            ["Sport", "Utility", "Sedan"]
        )
        
        # Base Policy
        base_policy = st.selectbox(
            "Base Policy",
            ["Liability", "Collision", "All Perils"]
        )
    
    with col2:
        st.header("Personal & Vehicle Details")
        
        # Vehicle Price
        vehicle_price = st.selectbox(
            "Vehicle Price Range",
            ["more than 69000", "20000 to 29000", "30000 to 39000", "less than 20000",
             "40000 to 59000", "60000 to 69000"]
        )
        
        # Days Policy Accident
        days_policy_accident = st.selectbox(
            "Days Between Policy Start and Accident",
            ["more than 30", "15 to 30", "none", "1 to 7", "8 to 15"]
        )
        
        # Days Policy Claim
        days_policy_claim = st.selectbox(
            "Days Between Policy Start and Claim",
            ["more than 30", "15 to 30", "8 to 15", "none"]
        )
        
        # Past Number of Claims
        past_number_of_claims = st.selectbox(
            "Past Number of Claims",
            ["none", "1", "2 to 4", "more than 4"]
        )
        
        # Age of Vehicle
        age_of_vehicle = st.selectbox(
            "Age of Vehicle",
            ["3 years", "6 years", "7 years", "more than 7", "5 years", "new", "4 years", "2 years"]
        )
        
        # Age of Policy Holder
        age_of_policy_holder = st.selectbox(
            "Age of Policy Holder",
            ["26 to 30", "31 to 35", "41 to 50", "51 to 65", "21 to 25", "36 to 40",
             "16 to 17", "over 65", "18 to 20"]
        )
        
        # Police Report Filed
        police_report_filed = st.selectbox(
            "Police Report Filed",
            ["No", "Yes"]
        )
        
        # Witness Present
        witness_present = st.selectbox(
            "Witness Present",
            ["No", "Yes"]
        )
        
        # Agent Type
        agent_type = st.selectbox(
            "Agent Type",
            ["External", "Internal"]
        )
        
        # Number of Supplements
        number_of_supplements = st.selectbox(
            "Number of Supplements",
            ["none", "more than 5", "3 to 5", "1 to 2"]
        )
        
        # Address Change Claim
        address_change_claim = st.selectbox(
            "Address Change from Claim",
            ["1 year", "no change", "4 to 8 years", "2 to 3 years", "under 6 months"]
        )
        
        # Number of Cars
        number_of_cars = st.selectbox(
            "Number of Cars",
            ["3 to 4", "1 vehicle", "2 vehicles", "5 to 8", "more than 8"]
        )
        
        st.header("Numerical Features")
        
        # Week of Month
        week_of_month = st.slider("Week of Month", 1, 5, 3)
        
        # Week of Month Claimed
        week_of_month_claimed = st.slider("Week of Month Claimed", 1, 5, 3)
        
        # Age
        age = st.slider("Age", 0, 80, 40)
        
        # Policy Number
        policy_number = st.number_input("Policy Number", 1, 15420, 7710)
        
        # Rep Number
        rep_number = st.slider("Rep Number", 1, 16, 8)
        
        # Deductible
        deductible = st.slider("Deductible", 300, 700, 400)
        
        # Driver Rating
        driver_rating = st.slider("Driver Rating", 1, 4, 2)
        
        # Year
        year = st.slider("Year", 1994, 1996, 1995)
    
    # Prediction button
    if st.button("Predict Fraud", type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Month': [month],
            'WeekOfMonth': [week_of_month],
            'DayOfWeek': [day_of_week],
            'Make': [make],
            'AccidentArea': [accident_area],
            'DayOfWeekClaimed': [day_of_week_claimed],
            'MonthClaimed': [month_claimed],
            'WeekOfMonthClaimed': [week_of_month_claimed],
            'Sex': [sex],
            'MaritalStatus': [marital_status],
            'Age': [age],
            'Fault': [fault],
            'PolicyType': [policy_type],
            'VehicleCategory': [vehicle_category],
            'VehiclePrice': [vehicle_price],
            'Days_Policy_Accident': [days_policy_accident],
            'Days_Policy_Claim': [days_policy_claim],
            'PastNumberOfClaims': [past_number_of_claims],
            'AgeOfVehicle': [age_of_vehicle],
            'AgeOfPolicyHolder': [age_of_policy_holder],
            'PoliceReportFiled': [police_report_filed],
            'WitnessPresent': [witness_present],
            'AgentType': [agent_type],
            'NumberOfSuppliments': [number_of_supplements],
            'AddressChange_Claim': [address_change_claim],
            'NumberOfCars': [number_of_cars],
            'Year': [year],
            'BasePolicy': [base_policy],
            'PolicyNumber': [policy_number],
            'RepNumber': [rep_number],
            'Deductible': [deductible],
            'DriverRating': [driver_rating]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            st.header("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ HIGH RISK - Potential Fraud Detected")
                else:
                    st.success("✅ LOW RISK - No Fraud Detected")
            
            with col2:
                st.metric("Fraud Probability", f"{prediction_proba[1]:.2%}")
                st.metric("No Fraud Probability", f"{prediction_proba[0]:.2%}")
            
            # Show confidence level
            confidence = max(prediction_proba)
            st.info(f"Model Confidence: {confidence:.2%}")
            
            # Show input summary
            with st.expander("Input Summary"):
                st.dataframe(input_data.T, column_config={"0": "Value"})
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()