import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st


def load_model():
    model = joblib.load('svr_binary.pkl')  
    scaler = joblib.load('scaler_svr_binary.pkl')  
    return model, scaler

def main():
    st.title("Iris Flower Prediction - SVR_Binary")

    st.subheader("Enter the Features for Prediction")
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.00, value=5.0)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

    user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    model, scaler = load_model()

    prediction_text = "" 

    if st.button("Get Prediction"):
        user_data_scaled = scaler.transform(user_data)

        prediction = model.predict(user_data_scaled)
        prediction_binary = 1 if prediction[0] >= 0.5 else 0 
        prediction_text = f"The predicted class is: {'Setosa' if prediction_binary == 0 else 'Versicolor'}"

    st.write(prediction_text)

if __name__ == "__main__":
    main()
