# Customer Churn Prediction

## Project Overview
This project uses Machine Learning to predict whether a telecom customer is likely to leave the service (churn).

## Dataset
- Source: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
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

The Logistic Regression model was evaluated on a held-out test dataset.

- Accuracy: 73.81%
- Churn precision: 0.50
- Churn recall: 0.78
- Churn F1-score: 0.61

Class weighting improved churn recall compared with the initial model, while overall accuracy decreased. The preferred model depends on the business cost of missed churn predictions versus unnecessary retention actions.

## Key Finding
In this dataset, month-to-month contract customers had a higher churn rate (42.71%) than one-year contract customers (11.27%) and two-year contract customers (2.83%).

## Future Improvements
- Improve churn recall.
- Compare multiple machine learning models.
- Build an interactive dashboard.
- Develop a simple interface for predictions.
## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the training script

```bash
python main.py
```

This trains the model, evaluates its performance, displays visualizations, and saves `churn_model.pkl`.

### 3. Run a sample prediction

After generating `churn_model.pkl`, run:

```bash
python predict.py
```

**Note:** The dataset file `WA_Fn-UseC_-Telco-Customer-Churn.csv` must be present in the project folder before running `main.py`. The trained model file must be present before running `predict.py`.
