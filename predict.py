#Customer ka churn predict karo#

import joblib
import pandas as pd

# Load the trained model
model = joblib.load("churn_model.pkl")

# Sample customer details
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 53.85,
    "TotalCharges": 108.15
}

# Convert customer details into a DataFrame
customer_df = pd.DataFrame([customer])

# Predict churn
prediction = model.predict(customer_df)[0]
probability = model.predict_proba(customer_df)[0][1]

if prediction == 1:
    print("Prediction: Customer may churn.")
else:
    print("Prediction: Customer is likely to stay.")

print(f"Estimated churn probability: {probability:.2%}")
#