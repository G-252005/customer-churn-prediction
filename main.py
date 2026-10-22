import joblib
import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


# Convert TotalCharges to numeric and handle missing values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("Dataset information:")
print(df.info())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCustomer Churn Count:")
print(df["Churn"].value_counts())

print("\nCustomer Churn Percentage:")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)

print("\nRows with missing TotalCharges:")

print(
    df.loc[
        df["TotalCharges"].isna(),
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

print("\nTenure summary:")
print(df["tenure"].describe())

print("\nChurn by Contract Type:")

churn_by_contract = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(churn_by_contract.round(2))
#

import matplotlib.pyplot as plt

churn_by_contract["Yes"].plot(
    kind="bar",
    title="Customer Churn Rate by Contract Type",
    ylabel="Churn Rate (%)",
    xlabel="Contract Type",
    rot=0
)

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report

# Features and target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify column types
categorical_cols = X.select_dtypes(include=["str", "object"]).columns
numerical_cols = X.select_dtypes(include=["number"]).columns

# Prepare numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)

# Create model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Compare the original and balanced models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

models = {
    "Original Logistic Regression": LogisticRegression(max_iter=1000),
    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
}

for name, classifier in models.items():
    comparison_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier)
    ])

    comparison_model.fit(X_train, y_train)
    predictions = comparison_model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Churn Precision:", round(precision_score(y_test, predictions), 3))
    print("Churn Recall:", round(recall_score(y_test, predictions), 3))
    print("Churn F1-score:", round(f1_score(y_test, predictions), 3))

# Keep the balanced model as the saved model
model.fit(X_train, y_train)
joblib.dump(model, "churn_model.pkl")
print("\nBalanced model saved successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"]
)

plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.show()

