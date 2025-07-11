# Customer Churn Prediction

## About This Project

This project develops a machine learning model to predict customer churn for a telecom company using the **IBM Telco Customer Churn Dataset**. The objective is to identify customers who are likely to leave the service, enabling the company to proactively engage retention strategies.

The project demonstrates the full ML pipeline, including data preprocessing, model training, evaluation, and deployment via a user-friendly Streamlit dashboard for real-time predictions on new data.

---

## Features

- **Data Preprocessing:** Handling missing values, encoding categorical variables, scaling numerical features.
- **Model Training:** Logistic Regression and Decision Tree classifiers.
- **Model Evaluation:** Metrics including accuracy, ROC-AUC, confusion matrix, and classification reports.
- **Feature Importance:** Analysis to understand key churn drivers.
- **Interactive Dashboard:** Streamlit app to upload new customer data and get churn predictions.
- **Model Persistence:** Saving and loading trained models for reuse.

---

## Dataset

- **Source:** IBM Telco Customer Churn Dataset (available on [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn))
- **Contents:** Customer demographics, account info, service usage, and churn labels.
- **Preprocessing:** Numeric conversion, missing value imputation, one-hot encoding, and feature scaling.
