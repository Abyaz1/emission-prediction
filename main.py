import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Title dan Deskripsi
st.title("Carbon Footprint Predictor")
st.markdown("""
This web predicts the carbon footprint (kg CO₂) based on user activity.
Enter your activity information below.
""")


# Input Data
st.header("Insert the data")

e = float(st.number_input("Electric consumption (kWh/month)", min_value=0, value=100))
i = float(st.number_input("Industry output", min_value=0, value=50))
t = float(st.number_input("Total Transportation Emission", min_value=0, value=100))
r = float(st.number_input("Residential output", min_value=0, value=5))

# Prediksi Jejak Karbon
# Menggunakan format float dengan titik
#prediction = -8270.1 + 1 * e + 1 * i + 1 * t + 1 * r

if st.button("Calculate"):
    prediction =  -8.2701 + 1*e + 1*i + 1*t + 1*r
    st.subheader(f"Prediksi Jejak Karbon Anda: {prediction:.2f} kg CO₂")
    if prediction > 500:
        st.warning("Jejak karbon Anda tinggi. Pertimbangkan untuk mengurangi konsumsi energi atau transportasi!")
    else:
        st.success("Jejak karbon Anda berada pada level yang cukup rendah. Pertahankan kebiasaan baik Anda!")

# Footer
st.markdown("""
---
Made with Streamlit to help raise environmental awareness🌍
""")

