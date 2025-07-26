import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

# Set page configuration
st.set_page_config(page_title="Employee Salary Predictor", page_icon="📊", layout="wide")

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
    }
    .stSlider .st-bx {
        background-color: #e8ecef;
        border-radius: 8px;
    }
    .stSelectbox {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 5px;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stMarkdown {
        color: #34495e;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center;'>📊 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Predict whether an employee earns >50K or ≤50K</p>", unsafe_allow_html=True)

# Function to map education to educational-num
def map_education_to_num(education):
    education_map = {
        "Bachelors": 13,
        "Masters": 14,
        "PhD": 16,
        "HS-grad": 9,
        "Assoc": 11,
        "Some-college": 10
    }
    return education_map.get(education, 9)  # Default to HS-grad if not found

# Function to encode and align DataFrame with model features
def preprocess_data(df, model):
    # Define categorical and numerical columns
    categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Get expected features from the model
    expected_features = model.feature_names_in_
    
    # Add missing columns with zeros
    for col in expected_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Remove extra columns not in expected features
    df_encoded = df_encoded[expected_features]
    
    return df_encoded

# Layout with columns
col1, col2 = st.columns([1, 2])

# Sidebar for inputs
with st.sidebar:
    st.markdown("<h2>🛠️ Employee Details</h2>", unsafe_allow_html=True)
    age = st.slider("Age", 18, 65, 30, help="Select the employee's age")
    education = st.selectbox("Education Level", [
        "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
    ], help="Choose the highest education level")
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked"
    ], help="Select the work class")
    occupation = st.selectbox("Job Role", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
        "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
        "Protective-serv", "Armed-Forces"
    ], help="Select the employee's occupation")
    hours_per_week = st.slider("Hours per Week", 1, 80, 40, help="Average hours worked per week")
    capital_gain = st.slider("Capital Gain", 0, 100000, 0, help="Capital gains in USD")
    capital_loss = st.slider("Capital Loss", 0, 5000, 0, help="Capital losses in USD")
    fnlwgt = st.slider("Final Weight", 10000, 1000000, 100000, help="Sampling weight")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select gender")
    marital_status = st.selectbox("Marital Status", [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse"
    ], help="Select marital status")
    relationship = st.selectbox("Relationship", [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried"
    ], help="Select relationship status")
    race = st.selectbox("Race", [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
        "Other", "Black"
    ], help="Select race")
    native_country = st.selectbox("Native Country", [
        "United-States", "Canada", "Mexico", "Philippines", "Germany",
        "Puerto-Rico", "India", "China", "Japan", "England", "Other"
    ], help="Select native country")

# Create input DataFrame
with col1:
    st.markdown("### 📋 Input Preview")
    input_df = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'fnlwgt': [fnlwgt],
        'educational-num': [map_education_to_num(education)],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'gender': [gender],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })
    st.dataframe(input_df, use_container_width=True)

    # Predict button
    if st.button("🔍 Predict Salary", use_container_width=True):
        try:
            # Preprocess the input DataFrame
            input_df_encoded = preprocess_data(input_df, model)
            prediction = model.predict(input_df_encoded)
            st.markdown(f"<h3 style='color: #27ae60;'>✅ Prediction: {prediction[0]}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Batch prediction section
with col2:
    st.markdown("### 📂 Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv", help="CSV must include columns: age, workclass, fnlwgt, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country")
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.markdown("**Uploaded Data Preview**")
        st.dataframe(batch_data.head(), use_container_width=True)
        try:
            # Preprocess the batch data
            batch_data_encoded = preprocess_data(batch_data, model)
            batch_preds = model.predict(batch_data_encoded)
            batch_data['PredictedClass'] = batch_preds
            st.markdown("**✅ Predictions**")
            st.dataframe(batch_data.head(), use_container_width=True)
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name='predicted_classes.csv',
                mime='text/csv',
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Batch prediction error: {str(e)}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Built with Streamlit | Powered by me</p>", unsafe_allow_html=True)
