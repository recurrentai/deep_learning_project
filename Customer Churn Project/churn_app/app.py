import streamlit as st
import pickle
import numpy as np
from keras.models import load_model

# Load trained model
model = load_model("ANN_model.keras")

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

# Take user input
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 18, 100)
tenure = st.slider("Tenure", 0, 10)
balance = st.number_input("Balance")
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = 1 if st.selectbox("Has Credit Card?", ["Yes", "No"]) == "Yes" else 0
is_active_member = 1 if st.selectbox("Is Active Member?", ["Yes", "No"]) == "Yes" else 0
estimated_salary = st.number_input("Estimated Salary")

# Encode geography and gender manually
geo_dict = {"France": 0, "Germany": 1, "Spain": 2}
gender_dict = {"Female": 0, "Male": 1}

input_data = np.array([[credit_score, geo_dict[geography], gender_dict[gender],
                        age, tenure, balance, num_of_products,
                        has_cr_card, is_active_member, estimated_salary]])

# Apply scaling
input_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    prediction = (prediction > 0.3)

    if prediction:
#        st.error("⚠️ Customer is likely to Exit!")
        st.image("Exit.jpg", caption="High Risk of Churn", use_column_width=True)
    else:
#       st.success("✅ Customer is likely to Stay.")
        st.image("happy.jpeg", caption="Customer Retained", use_column_width=True)
