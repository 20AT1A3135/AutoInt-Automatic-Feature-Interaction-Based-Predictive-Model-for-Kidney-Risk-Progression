
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle as pkl

ct = pkl.load(open('ct.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))


labels = {
    0: 'High,\nSYMPTOMS: Pain in lower back,\nRestless leg syndrome',
    1: 'Moderate,\nSYMPTOMS: Urinary tract infections',
    2: 'Normal,\nSYMPTOMS: Swelling in your hands or feet',
}

for key, value in labels.items():
    print(value.replace("\n", "\n"))



# # Load the trained model from the saved weights
# auto_int_model = tf.keras.models.load_model('/content/auto_int_weights.h5')

# # Define and initialize label encoders for each categorical column
# label_encoders = {
#     'gender': LabelEncoder(),
#     'comorbidities_myocardial_infarction': LabelEncoder(),
#     'comorbidities_congestive_heart_failure': LabelEncoder(),
#     'comorbidities_chronic_kidney_disease': LabelEncoder()
# }

# Function to preprocess input data
def preprocess_data(data):
    # Encode categorical columns
    # for col in label_encoders:
    #     if col in data.columns:  # Check if the column exists in the DataFrame
    #         data[col] = label_encoders[col].fit_transform(data[col])

    # # Other preprocessing steps...

    # return data
    return ct.transform(data)

# Function to predict risk level
def predict_risk_level(data):
    # Add your prediction logic here...
    return model.predict(data)

# Create the web application using Streamlit
def main():
    st.title('Kidney Transplant Risk Prediction')
    st.write('Enter patient information to predict the risk level.')

    # Create input fields for patient information
    age = st.number_input('Age at Nephrectomy', min_value=0, max_value=120, value=50)
    gender = st.selectbox('Gender', ['male', 'female'])
    bmi = st.number_input('Body Mass Index', min_value=0.0, value=33.71)
    myocardial_infarction = st.checkbox('Myocardial Infarction')
    congestive_heart_failure = st.checkbox('Congestive Heart Failure')
    chronic_kidney_disease = st.checkbox('Chronic Kidney Disease')
    last_preop_egfr = st.number_input('Last Preop eGFR Value', min_value=0, value=67)
    first_postop_egfr = st.number_input('First Postop eGFR Value', min_value=0, value=58)
    last_postop_egfr = st.number_input('Last Postop eGFR Value', min_value=0, value=56)
    preop_to_postop_change = st.number_input('Preop to Postop Change', value=-9)
    postop_to_last_change = st.number_input('Postop to Last Change', value=-2)

    # Create a DataFrame for the single patient's data
    patient_data = pd.DataFrame({
        'age_at_nephrectomy': [age],
        'gender': [gender],
        'body_mass_index': [bmi],
        'comorbidities_myocardial_infarction': [myocardial_infarction],
        'comorbidities_congestive_heart_failure': [congestive_heart_failure],
        'comorbidities_chronic_kidney_disease': [chronic_kidney_disease],
        'last_preop_egfr_value': [last_preop_egfr],
        'first_postop_egfr_value': [first_postop_egfr],
        'last_postop_egfr_value': [last_postop_egfr],
        'preop_to_postop_change': [preop_to_postop_change],
        'postop_to_last_change': [postop_to_last_change]
    })

    # Preprocess the input data
    preprocessed_data = preprocess_data(patient_data)

    if st.button('Predict Risk Level'):
        # Predict the risk level
        risk_level = predict_risk_level(preprocessed_data)
        st.write(f'Predicted Risk Level: {labels[risk_level[0]]}')

if __name__ == '__main__':
    main()
