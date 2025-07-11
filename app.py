import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model once at start
model = load('logistic_regression_model.joblib')

st.title("Customer Churn Prediction Dashboard")

st.write("Upload your customer data CSV to get churn predictions:")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV into dataframe
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    # Preprocessing steps - you must replicate your earlier preprocessing here!
    # Example: handle missing TotalCharges, encode categorical, scale numeric features, etc.

    # Example preprocessing (adjust based on your actual preprocessing)
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)

    # Drop columns not used for prediction (e.g. 'customerID' if present)
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)

    # Encoding categorical variables
    # You need to ensure the uploaded data has the same columns/features as model expects
    cat_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Scaling numeric features - assuming scaler fitted during training is saved too
    # If you saved your scaler, load it here and transform
    # For demo, skipping scaling or you can use the same scaler saved with joblib
    # scaler = load('scaler.joblib')
    # df[num_cols] = scaler.transform(df[num_cols])

    # Align columns of uploaded data with model input features (handle missing columns)
    model_features = model.feature_names_in_  # scikit-learn 1.0+
    for col in model_features:
        if col not in df.columns:
            df[col] = 0  # add missing columns with 0s
    df = df[model_features]

    # Predict churn probability
    churn_probs = model.predict_proba(df)[:, 1]
    churn_preds = model.predict(df)

    # Display results
    result_df = df.copy()
    result_df['Churn_Probability'] = churn_probs
    result_df['Churn_Prediction'] = churn_preds

    st.write("Prediction Results:")
    st.dataframe(result_df[['Churn_Probability', 'Churn_Prediction']].head())

    # Simple visualization
    st.bar_chart(result_df['Churn_Probability'])
