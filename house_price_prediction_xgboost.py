from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import streamlit as st
import numpy as np
import xgboost as xgb

# loading data
#--------------
df = pd.read_csv("train.csv")
#--------------
# numeric features
feature_num = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']

# string features
feature_st = ['HouseStyle', 'RoofStyle']

# Target
target = 'SalePrice'

x = df[feature_num + feature_st]
y = df[target]

#-------------
# split data
#-------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#-----------------
# preparing data
#----------------
preprocess = ColumnTransformer(transformers=[
                ('nume', StandardScaler(), feature_num),
                ('cat', OneHotEncoder(handle_unknown="ignore"), feature_st)
                ]
                               )
#---------------
# model
model = xgb.XGBRegressor(objective='reg:squarederror',
                        n_estimators=500, 
                        learning_rate = 0.03,
                        random_state=42,
                        max_depth= 5)
# links the steps(preprcessing data, model)
models = Pipeline([
    ('preprocess', preprocess),
    ('model', model)
])

# Training model
models.fit(x_train, y_train)

#----------------------------
# test the model
#----------------------------

st.title("House Price Prediction")

col1, col2 = st.columns(2)
with col1:
    oq = st.slider("OverallQual", min_value=1, max_value=10, value = 5)
with col2:
    gla = st.slider("GrLivArea", min_value=1, max_value=10000, value = 2000)

col3, col4 = st.columns(2)
with col3:
    gc =st.slider("GarageCars", min_value=0, max_value=10, value = 2)
with col4:
   tb = st.slider("TotalBsmtSF", min_value=1, max_value=10000, value = 2000)

col5, col6 =st.columns(2)
with col5:
    hs = st.selectbox("HouseStyle", ['1Story', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'])
with col6:
    rs = st.selectbox("RoofStyle", ['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'])
    
result = pd.DataFrame([[oq, gla, gc, tb, hs, rs]], columns=[ 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF','HouseStyle',  'RoofStyle',])
res = models.predict(result)[0]

def print_result():
    st.write("prediction result: ", res)
if st.button("predict"):
    print_result()