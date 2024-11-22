import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("data/hepatitis_eda.csv")

# Feature selection and preprocessing
label_encoder_country = LabelEncoder()
df['Country'] = label_encoder_country.fit_transform(df['Country'])
label_encoder_gender = LabelEncoder()
df['Gender'] = label_encoder_gender.fit_transform(df['Gender'])

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
df['Age'] = df['Age'].map(age_mapping)

'''label_encoder_age = LabelEncoder()
df['Age'] = label_encoder_age.fit_transform(df['Age'])'''


# Select features and targets
X = df[['Country', 'Time', 'Gender', 'Age']]
y_chronic = df['Chronic_percentage']  
y_acute = df['Acute_percentage']  

# Train-test split
X_train, X_test, y_train_chronic, y_test_chronic = train_test_split(X, y_chronic, test_size=0.2, random_state=42)
X_train, X_test, y_train_acute, y_test_acute = train_test_split(X, y_acute, test_size=0.2, random_state=42)

# Create and train Random Forest model for Chronic_percentage
rf_chronic = RandomForestRegressor(n_estimators=100, random_state=42)
rf_chronic.fit(X_train, y_train_chronic)

# Create and train Random Forest model for Acute_percentage
rf_acute = RandomForestRegressor(n_estimators=100, random_state=42)
rf_acute.fit(X_train, y_train_acute)

# Evaluate models
chronic_preds = rf_chronic.predict(X_test)
acute_preds = rf_acute.predict(X_test)

# Calculate root mean squared error (RMSE)
rmse_chronic = np.sqrt(mean_squared_error(y_test_chronic, chronic_preds))
rmse_acute = np.sqrt(mean_squared_error(y_test_acute, acute_preds))

print(f'RMSE for Chronic_percentage: {rmse_chronic}')
print(f'RMSE for Acute_percentage: {rmse_acute}')


# Save models and encoders
models = {
    'chronic': rf_chronic,
    'acute': rf_acute
}

# Salva il dizionario in un unico file
joblib.dump(models, 'models/models_rf.pkl')

joblib.dump(label_encoder_country, "models/label_encoder_country.pkl")
joblib.dump(label_encoder_gender, "models/label_encoder_gender.pkl")
#joblib.dump(label_encoder_age, "models/label_encoder_age.pkl")


