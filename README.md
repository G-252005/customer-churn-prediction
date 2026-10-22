# Customer Churn Prediction

## Project Overview
This project uses Machine Learning to predict whether a telecom customer is likely to leave the service (churn).

## Dataset
- Dataset: Telco Customer Churn
- Records: 7,043 customers
- Target column: Churn

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Logistic Regression

## Project Steps
1. Loaded and explored the dataset.
2. Converted TotalCharges into numeric format.
3. Handled missing values.
4. Analysed churn rates by contract type.
5. Visualized churn rates using a bar chart.
6. Prepared categorical and numerical features.
7. Trained a Logistic Regression model.
8. Evaluated the model using accuracy, precision, recall, F1-score and a confusion matrix.
9. Saved the trained model as `churn_model.pkl`.

## Model Results
- Test accuracy: approximately 80.55%
- Churn precision: 0.66
- Churn recall: 0.56
- Churn F1-score: 0.60

## Key Finding
In this dataset, month-to-month contract customers had a higher churn rate (42.71%) than one-year contract customers (11.27%) and two-year contract customers (2.83%).

## Future Improvements
- Improve churn recall.
- Compare multiple machine learning models.
- Build an interactive dashboard.
- Develop a simple interface for predictions.