import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Title of the app
st.title('Breast Cancer Prediction')

# Load Dataset
df = pd.read_csv('data.csv')
encoder =  LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

# Split Data into Features and Target Variable
x = df.drop(['Unnamed: 32', 'id', 'diagnosis'], axis=1)
y = df['diagnosis']

# Split Dataset into Train and Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

# Train Models
model1 = LogisticRegression()
model1.fit(x_train, y_train)
model2 = SVC(kernel='linear')
model2.fit(x_train, y_train)

# Add Input Fields for User to Enter Their Own Data
st.sidebar.header("Enter the features for prediction")
input_data = []
for i, column in enumerate(x.columns):
    value = st.sidebar.slider(f"Enter {column}", float(x[column].min()), float(x[column].max()))
    input_data.append(value)

# Convert Input Data into a 2D Array for Prediction
input_array = np.asarray(input_data).reshape(1, -1)
input_array_scaled = scalar.transform(input_array)

# Make Predictions with Both Models
prediction1 = model1.predict(input_array_scaled)
prediction2 = model2.predict(input_array_scaled)

# Display Prediction Results
label = {0: 'Benign', 1: 'Malignant'}
st.write(f"🧠 Logistic Regression Prediction: {label[prediction1[0]]}")
st.write(f"🤖 SVM (Linear Kernel) Prediction: {label[prediction2[0]]}")

# Display Model Accuracy
st.write(f"Model 1 (Logistic Regression) Accuracy: {accuracy_score(y_test, model1.predict(x_test))}")
st.write(f"Model 2 (SVM) Accuracy: {accuracy_score(y_test, model2.predict(x_test))}")
st.write("The accuracy and the results of the models may vary based on the dataset and the features used. "
         "Make sure to enter valid values for the features.")
