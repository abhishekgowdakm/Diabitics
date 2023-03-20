import streamlit as st
import pandas as pd
import tensorflow as tf;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import MinMaxScaler;
import seaborn as sns;
import numpy as np;
from PIL import Image;

diabetics = Image.open('diabetics.jpg')
walk = Image.open('walk.webp')

data = pd.read_csv('./diabetes.csv')

st.header("Diabetes Prediction through various paramenter")

st.subheader("Using Tensorflow")

st.bar_chart(data)

X = data.drop('Outcome',axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

mn = MinMaxScaler()

X_train = mn.fit_transform(X_train)
X_test = mn.fit_transform(X_test)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(8,activation='relu'))
model.add(tf.keras.layers.Dense(8,activation='relu'))
model.add(tf.keras.layers.Dense(8,activation='relu'))
model.add(tf.keras.layers.Dense(2))

model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=300)


def user_report():
  pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
  glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  bp = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  bmi = st.sidebar.slider('BMI', 0,67, 20 )
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'pregnancies':pregnancies,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_value = user_report()

predict = model.predict(user_value)

pred = np.argmax(predict[0])

st.header("Results")
if pred == 0:
  st.write("Your healthy and not diabitic")
  st.image(walk)
else:
  st.write("Diabiatics and please contact doctor for treatment")
  st.image(diabetics)