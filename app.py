import pandas as pd
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title('🌸 Iris Flower Prediction')

# Sliders with correct ranges
sepal_length = st.slider('Sepal Length', 4.3, 7.9, 5.4)
sepal_width  = st.slider('Sepal Width', 2.0, 4.4, 3.4)
petal_length = st.slider('Petal Length', 1.0, 6.9, 1.3)
petal_width  = st.slider('Petal Width', 0.1, 2.5, 0.2)

# Prepare input
input_data = pd.DataFrame({
    'sepal_length':[sepal_length],
    'sepal_width':[sepal_width],
    'petal_length':[petal_length],
    'petal_width':[petal_width]
})

# Scale
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

# Get class index
predicted_class = np.argmax(prediction)

species = ['setosa', 'versicolor', 'virginica']

st.write("### Prediction Probabilities:")
st.write(prediction)

st.write(f"## 🌼 Predicted Flower: {species[predicted_class]}")