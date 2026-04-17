# House Price Prediction (XGBoost + ML Pipeline + Streamlit)

A machine learning web application that predicts house prices using XGBoost with a full preprocessing pipeline.  
The project combines data preprocessing and modeling into a single pipeline and provides a user interface using Streamlit.

---

## Features

- Full ML pipeline (preprocessing + model)
- Uses XGBoost Regressor for prediction
- Handles numerical and categorical features
- Feature scaling and encoding included
- Interactive UI using Streamlit

---

## How It Works

1. Load dataset using pandas  
2. Select features:
   - Numerical:
     - OverallQual
     - GrLivArea
     - GarageCars
     - TotalBsmtSF  
   - Categorical:
     - HouseStyle
     - RoofStyle  
3. Split data into training and testing sets  
4. Preprocessing:
   - StandardScaler for numerical features  
   - OneHotEncoder for categorical features  
5. Build pipeline:
   - Preprocessing → XGBoost model  
6. Train model  
7. User inputs data via Streamlit  
8. Pipeline processes input and predicts house price  

---

## Input Features

- Overall Quality
- Ground Living Area
- Garage Capacity
- Basement Area
- House Style
- Roof Style

---

## Model Details

- Model: XGBRegressor  
- Objective: reg:squarederror  
- Number of estimators: 500  
- Learning rate: 0.03  
- Max depth: 5  
- Random state: 42  

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
