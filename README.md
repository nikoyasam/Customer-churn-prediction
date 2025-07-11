Customer Churn Prediction
Project Overview
This project builds a machine learning model to predict customer churn for a telecom company using the IBM Telco Customer Churn Dataset. The goal is to identify customers who are likely to leave the service, enabling the business to take proactive retention actions.

Features
Data cleaning and preprocessing including handling missing values and encoding categorical variables.

Model training using Logistic Regression and Decision Tree classifiers.

Model evaluation using accuracy, ROC-AUC, confusion matrix, and classification reports.

Feature importance analysis to understand key drivers of churn.

Streamlit dashboard for uploading new data and getting churn predictions interactively.

Saved model for easy deployment and reuse.

Dataset
Source: IBM Telco Customer Churn Dataset (publicly available on Kaggle and IBM)

Data Description: Includes customer demographics, account information, service usage, and churn label.

Preprocessing: Numeric conversion, missing value imputation, one-hot encoding, feature scaling.

How to Run

2. Training the Model
Run the notebook customer_churn_model.ipynb to:

Load and preprocess data

Train and evaluate models

Save the trained model as logistic_regression_model.joblib

3. Using the Streamlit Dashboard

Run the app to upload new customer data CSV and get churn predictions:

streamlit run app.py


File Structure

.
├── customer_churn_model.ipynb       # Jupyter notebook with model training code
├── logistic_regression_model.joblib # Saved trained model
├── app.py                          # Streamlit dashboard app             
├── README.md                      # Project documentation
└── data                           # Folder for datasets

