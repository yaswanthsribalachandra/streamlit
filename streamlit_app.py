import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title
st.title("📊 Simple Linear Regression App")

# Sidebar inputs
st.sidebar.header("Input Data")

# User input for X and Y
num_points = st.sidebar.slider("Number of Data Points", 10, 100, 20)

# Generate random data
np.random.seed(42)
X = np.random.rand(num_points, 1) * 10
y = 2.5 * X + np.random.randn(num_points, 1) * 2

# Show dataset
st.subheader("📂 Generated Dataset")
df = pd.DataFrame({"X": X.flatten(), "y": y.flatten()})
st.write(df)

# Train model
model = LinearRegression()
model.fit(X, y)

# Prediction
st.subheader("🔮 Make Prediction")
input_value = st.number_input("Enter X value", value=5.0)
prediction = model.predict([[input_value]])

st.write(f"Predicted y: {prediction[0][0]:.2f}")

# Plot
st.subheader("📈 Regression Line")
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Data Points')
ax.plot(X, model.predict(X), color='red', label='Regression Line')
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()

st.pyplot(fig)
