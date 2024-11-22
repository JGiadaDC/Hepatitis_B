# HEPATITIS B RISK PREDICTION APP (EUORPE)

### Description 
This application predicts the risk of contracting Hepatitis B  (in its chronic and acute forms) in Europe, based on factors such as country of residence, gender, age group, and year. It is designed to provide predictive estimates using a machine learning model trained with epidemiological data.

### Key features
- Prediction of chronic Hepatitis B risk.
- Prediction of acute Hepatitis B risk.
- Interactive interface built with Streamlit.
- Ability to make future predictions (up to the year 2050).

### Ethical considerations 
This project is intended solely for educational and training purposes and does not aim to provide accurate real-world predictions. The raw data used in this project was downloaded from the [ECDC Atlas](https://atlas.ecdc.europa.eu/public/index.aspx).
.

Due to inconsistencies in data availability across gender and age categories, interpolation was applied during the creation of the final dataset. Additionally, missing data for some countries were replaced using averages or similar methods to fill gaps. These adjustments may impact the accuracy and reliability of the predictions.

Despite these limitations, the project serves as a valuable example of how to train a regression model based on historical data while addressing challenges related to incomplete datasets.


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
```

2. Navigate into the project directory

```bash
cd Hepatitis_B
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
```

3. Install Dependencies
Install the required Python packages listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

4. Download Necessary Files
Some files, such as the .pkl models, are included in the models folder. 

### Usage 
To run the Streamlit application, use the command:
```bash
streamlit run src/app.py
```

- Follow the on-screen instructions to input data (country, gender, age, year).
- View the risk predictions on the results screen.

### Project structure
```bash
Hepatitis_B/
│
├── data/                             # Contains datasets and related input/output files
├── models/                           # Pre-trained models and encoders (.pkl files)
│   ├── models_rf.pkl                 # Random Forest model for predictions
│   ├── label_encoder_country.pkl     # Encoder for country categorization
│   └── label_encoder_gender.pkl      # Encoder for gender categorization
├── notebooks/                        # Jupyter Notebooks for analysis and prototyping
├── src/                              # Source code for the Streamlit app and Python modules
├── .gitignore                        # Specifies files or directories to exclude from Git
├── README.md                         # Project description and documentation
└── requirements.txt                  # Lists dependencies required to run the project

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

