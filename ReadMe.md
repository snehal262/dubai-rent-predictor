🏙️ Dubai Apartment Rent Predictor

An end-to-end machine learning project that predicts annual apartment rent in Dubai using property features, with an interactive Streamlit app for predictions, market exploration, and model explainability.

🔗 Live App: [Add your Streamlit Cloud link here]

📌 Overview

This project covers the full data science lifecycle:

Data Cleaning — handling invalid entries, outlier removal, feature engineering
Exploratory Data Analysis — understanding rent distributions and key drivers
Model Training — XGBoost regression with log-transformed target
Explainability — SHAP values to interpret individual predictions
Deployment — interactive Streamlit web app

Dataset: Dubai Properties - Apartments (Kaggle) — ~75,000 apartment listings scraped from a Dubai real estate portal.

🎯 Problem Statement

Rent prices in Dubai vary enormously by location, size, and property type, making it hard for renters, landlords, and analysts to quickly estimate a fair market rent. This project builds a predictive model that estimates annual rent based on property characteristics, and an app that makes this accessible to non-technical users.

🧹 Data Cleaning

Key decisions (full reasoning in model.py and 02_eda.ipynb):

Removed listings with Rent = 0 (data entry errors)
Floored rent at AED 10,000 — IQR-based lower bound went negative due to right-skewed distribution, so a domain-informed floor was used instead
Capped rent at the 99th percentile (AED 1,000,000) — values above this were villas/penthouses mislabelled as apartments, confirmed by manual inspection
Filtered to rental listings only (removed for-sale properties)
Standardised categorical fields (Furnishing, Type, Frequency)
Engineered features: Beds_x_Baths (interaction term), log_Rent (log-transformed target), Is_furnished (binary flag)

Result: 73,072 clean rows from ~75,000 raw listings.

📊 Exploratory Data Analysis

Key findings:

Rent distribution is heavily right-skewed (skewness > 1) → log transformation applied to the target
Area_in_sqft and Location are the strongest predictors of rent
Furnished apartments command a measurable premium over unfurnished
Significant rent variation across Dubai neighbourhoods — Downtown Dubai, Palm Jumeirah, and Dubai Marina rank among the most expensive

See 02_eda.ipynb for full charts and analysis.

🤖 Model

Algorithm: XGBoost Regressor
Target: log(Rent) — reverse-transformed to AED for predictions

Features used:

Area_in_sqft, Beds, Baths, Beds_x_Baths
Location, Furnishing, Type, Frequency (label-encoded)

Hyperparameters:

pythonXGBRegressor(
n_estimators=500,
max_depth=6,
learning_rate=0.05,
subsample=0.8,
colsample_bytree=0.8,
random_state=42
)

Performance (test set, 14,615 rows)

MetricValueR²0.906MAPE18.2%RMSEAED 53,030

The model explains 90.6% of variance in rent prices. The remaining error reflects factors not captured in the data — floor level, view, renovation quality, and landlord pricing strategy.

🧠 Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values were computed to understand:

Global feature importance — which features drive predictions across the dataset overall
Per-prediction explanations — for any single prediction, which features pushed the estimate up or down, and by how much

This makes the model interpretable rather than a black box — critical for real estate applications where stakeholders need to understand why a price was estimated.

💻 The App

App link: https://dubai-rent-predictor-inbrfcjxscydvmayrapfnt.streamlit.app/
Built with Streamlit, the app has three tabs:

🔮 Rent Predictor

Enter property details (location, beds, baths, area, furnishing, type) and get an instant rent estimate with a confidence range, monthly equivalent, and comparison against similar listings in the same area.

🗺️ Map View

Interactive map of Dubai showing median rent by neighbourhood, with filters by bedroom count and listing density.

🧠 Model Explainer

Global SHAP feature importance, plus a live waterfall-style breakdown of why the model arrived at a specific prediction for the property you configured.

🛠️ Tech Stack

Python — pandas, NumPy, scikit-learn
Modelling — XGBoost
Explainability — SHAP
Visualisation — Matplotlib, Seaborn, Plotly
App — Streamlit

🚀 Running Locally

bashgit clone https://github.com/<your-username>/dubai-rent-predictor.git
cd dubai-rent-predictor

conda create -n realestate python=3.11
conda activate realestate
pip install -r requirements.txt

# Train the model (optional — pkl files already included)

python model.py

# Run the app

streamlit run app.py

📁 Project Structure

dubai-rent-predictor/
├── app.py # Streamlit app
├── model.py # Data prep, training, SHAP, exports pkl files
├── 02_eda.ipynb # Exploratory data analysis
├── dubai_clean.csv # Cleaned dataset
├── model.pkl # Trained XGBoost model
├── encoders.pkl # Label encoders for categorical features
├── shap_explainer.pkl # SHAP TreeExplainer
├── feature_cols.pkl # Feature column order/metadata
├── requirements.txt
└── README.md

👤 Author

Snehal Ashlyn D Souza
LinkedIn | Email
