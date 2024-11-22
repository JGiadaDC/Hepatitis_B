import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression


# Load models and encoders
models = joblib.load("models/models_rf.pkl")

label_encoder_country = joblib.load("models/label_encoder_country.pkl")
label_encoder_gender = joblib.load("models/label_encoder_gender.pkl")
#label_encoder_age = joblib.load("models/label_encoder_age.pkl")
age_mapping = {
    '0-4': 0,
    '15-19': 2,
    '20-24': 3,
    '25-34': 4,
    '35-44': 5,
    '45-54': 6,
    '5-14': 1,
    '55-64': 7,
    '65+': 8
}


# Future predictions
def predict_future(chronic_model, acute_model, input_data, year):
    max_year_in_data = 2024  
    if year > max_year_in_data:
        known_years = pd.DataFrame({'Time': range(2000, max_year_in_data + 1)})
        chronic_preds = []
        acute_preds = []

        for y in known_years['Time']:
                    input_data['Time'] = y
                    chronic_preds.append(chronic_model.predict(input_data)[0])
                    acute_preds.append(acute_model.predict(input_data)[0])


        
        chronic_trend_model = LinearRegression()
        acute_trend_model = LinearRegression()
        chronic_trend_model.fit(known_years, chronic_preds)
        acute_trend_model.fit(known_years, acute_preds)

        # Predict future 
        chronic_pred = chronic_trend_model.predict([[year]])[0]
        acute_pred = acute_trend_model.predict([[year]])[0]

    else:
        # Predict for years in the dataset
        chronic_pred = chronic_model.predict(input_data)[0]
        acute_pred = acute_model.predict(input_data)[0]
    # avoid negative values
    chronic_pred = max(0, chronic_pred)
    acute_pred = max(0, acute_pred)
    
    return chronic_pred, acute_pred


# Function to make predictions
def make_prediction(gender, age, country, year):
    # Codify categorical values
    gender_encoded = label_encoder_gender.transform([gender])[0]
    #age_category_encoded = label_encoder_age.transform([age_category])[0]
    country_encoded = label_encoder_country.transform([country])[0]
    # Convert age using the predefined mapping
    age_encoded = age_mapping.get(age, -1)  # Use -1 if age is not found in the mapping

    # Prepare input data
    input_data = pd.DataFrame({
        'Country': [country_encoded],
        'Time': [year], 
        'Gender': [gender_encoded],
        'Age': [age_encoded]
    })

    # Models for prediction
    rf_chronic = models['chronic']
    rf_acute = models['acute']

    # Predicting
    chronic_pred, acute_pred = predict_future(rf_chronic, rf_acute, input_data, year)

    return chronic_pred, acute_pred


# Streamlit
st.title("Calculate your risk of contracting Hepatitis B in its Chronic and Acute phases")

# Input
country = st.selectbox("Select Country:", ['Croatia', 'Cyprus','Czechia','Denmark','Estonia', 'Finland','France',
                                           'Germany','Greece','Hungary','Iceland','Ireland','Italy','Latvia','Liechtenstein',
                                           'Lithuania','Luxembourg','Malta','Netherlands','Norway','Poland','Portugal',
                                           'Romania','Slovakia','Slovenia','Spain','Sweden','United Kingdom'])
gender = st.selectbox("Select gender:", ["Female", "Male"])
age = st.selectbox("Select age:", ["0-4", "5-14", "15-19", "20-34", "35-44", "45-54", "55-64", "65+"])
year = st.number_input('Select year:', min_value=2000, max_value=2050, value=2024)

# Button to calculate risk
if st.button("Calculate risk (%)"):
    chronic_pred, acute_pred = make_prediction(gender, age, country, year)

    # Show the result
    st.subheader("Prediction results:")
    st.write(f"Risk of getting Chronic phase: **{chronic_pred:.2f}%**")
    st.write(f"Risk of getting Acute phase: **{acute_pred:.2f}%**")



