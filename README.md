# HEPATITIS B RISK PREDICTION APP

### Description 
This application predicts the risk of contracting Hepatitis B (in its chronic and acute forms) based on factors such as country of residence, gender, age group, and year. It is designed to provide predictive estimates using a machine learning model trained with epidemiological data.

### Key features
- Prediction of chronic Hepatitis B risk.
- Prediction of acute Hepatitis B risk.
- Interactive interface built with Streamlit.
- Ability to make future predictions (up to the year 2050).

### System requirements
. Python 3.9 or higher
- Python libraries:
   - streamlit
   - pandas
   - scikit-learn
   - joblib
- Files:
   - models_rf.pkl (compressed machine learning models)
   - Encoders for categorical variables (gender, country, age)

### Installation 
Follow these steps to install the project on your computer:

1. Clone the repository:  

```bash
git clone https://github.com/JGiadaDC/Hepatitis_B.git

# Navigate into the project directory
cd Hepatitis_B
```

3. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Ensure that the required .pkl files are in the main directory.

### Usage 
To run the Streamlit application, use the command:
```bash
streamlit run app.py
```

- Follow the on-screen instructions to input data (country, gender, age, year).
- View the risk predictions on the results screen.

### Project structure
```bash
Hepatitis_B/
├── app.py                  # Code for the Streamlit application
├── model_training.py       # Code for training the ML model
├── models_rf.pkl           # Saved machine learning model
├── label_encoder_country.pkl  # Encoder for categorical variables
├── label_encoder_gender.pkl
├── label_encoder_age.pkl
├── requirements.txt        # Dependency list
├── README.md               # Project documentation
├── data/                   # Datasets used for training
│   ├── hepatitis_data.csv
│   └── eda/                # Exploratory data analysis
└── results/                # Model outputs and analyses
```

### Examples 
**Input:**
Country: Italy
Gender: Male
Age: 20-34
Year: 2030

**Output:**
Chronic Hepatitis B Risk: 12.5%
Acute Hepatitis B Risk: 8.3%

### Contact
**Author:** Giada
**Email:** giada@decarlo#gmail.com
**GitHub:** JGiadaDC

