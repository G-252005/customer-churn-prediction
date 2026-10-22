import joblib
import pandas as pd

model = joblib.load("churn_model.pkl")

print("=== Customer Churn Prediction ===")

tenure = int(input("Customer tenure (months): "))
monthly_charges = float(input("Monthly charges: "))
total_charges = float(input("Total charges: "))

contract = input(
    "Contract (Month-to-month, One year, Two year): "
)

internet_service = input(
    "Internet service (DSL, Fiber optic, No): "
)

payment_method = input(
    "Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)): "
)

customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet_service,
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

customer_df = pd.DataFrame([customer])

prediction = model.predict(customer_df)[0]
probability = model.predict_proba(customer_df)[0][1]

print("\nPrediction:", "May Churn" if prediction == 1 else "Likely to Stay")
print(f"Estimated churn probability: {probability:.2%}")
