# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:35:26 2024

@author: shubham
"""

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.title("Titanic Survival Predictor")


pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age")

# Create a DataFrame from user input
user_data = {'Pclass': pclass, 'Sex': sex, 'Age': age}  # Update with other features
user_df = pd.DataFrame([user_data])


categorical_features = ['Sex']  
cat_pclass = ['Pclass']
cat_age = ['Age']
le = LabelEncoder()
user_df[categorical_features] = le.fit_transform(user_df[categorical_features])
user_df[cat_pclass] = le.fit_transform(user_df[cat_pclass])
user_df[cat_age] = le.fit_transform(user_df[cat_age])


import pickle  # For model loading

# Load the trained model from a file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
    # Make predictions using the preprocessed user data
prediction = model.predict(user_df)[0]

if prediction == 1:
    st.write("Passenger is likely to survive.")
else:
    st.write("Passenger is likely to not survive.")