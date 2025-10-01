# HW1: Simple Linear Regression with CRISP-DM
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------------- Title ----------------
st.title("HW1: Simple Linear Regression App (y = ax + b + noise)")

st.write("""
This app demonstrates **Linear Regression** with synthetic data,  
following the **CRISP-DM process**.
""")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Data Generation Parameters")

a = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
b = st.sidebar.slider("Intercept (b)", min_value=-10.0, max_value=10.0, value=5.0, step=0.1)
noise = st.sidebar.slider("Noise Std Dev", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
n = st.sidebar.slider("Number of points (n)", min_value=20, max_value=500, value=100, step=10)

# ---------------- CRISP-DM Explanation ----------------
st.header("CRISP-DM Process")

with st.expander("1. Business Understanding"):
    st.write("""
    Goal: Simulate a simple dataset (y = ax + b + noise) and fit a linear regression model.  
    Success: The app should allow user to adjust parameters (a, b, noise, n)  
    and visualize how regression performs.
    """)

with st.expander("2. Data Understanding"):
    st.write("""
    The generated data is synthetic: X ranges from 0 to 10,  
    y is computed as y = ax + b + noise.  
    Noise follows a Gaussian distribution.
    """)

with st.expander("3. Data Preparation"):
    st.write("""
    - Generate evenly spaced X values.  
    - Add Gaussian noise to y.  
    - Reshape arrays into suitable format for sklearn.
    """)

with st.expander("4. Modeling"):
    st.write("""
    - Model: Linear Regression (sklearn).  
    - Fit model on (X, y).  
    - Predict y_hat.
    """)

with st.expander("5. Evaluation"):
    st.write("""
    Evaluate performance using:  
    - Mean Squared Error (MSE)  
    - R-squared (R²)  
    """)

with st.expander("6. Deployment"):
    st.write("""
    This app is deployed via **Streamlit**,  
    enabling interactive parameter adjustment and real-time visualization.
    """)

# ---------------- Data Generation ----------------
X = np.linspace(0, 10, n).reshape(-1, 1)
y_true = a * X + b
y = y_true + np.random.normal(0, noise, size=(n, 1))

# ---------------- Modeling ----------------
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# ---------------- Metrics ----------------
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.subheader("Model Performance Metrics")
st.write(f"True equation: y = {a}x + {b} + noise(std={noise})")
st.write(f"Fitted model: y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}")
st.write(f"Mean Squared Error (MSE): **{mse:.2f}**")
st.write(f"R-squared (R²): **{r2:.4f}**")

# ---------------- Visualization ----------------
st.subheader("Visualization")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, label="Data (with noise)", alpha=0.6)
ax.plot(X, y_true, label="True function (no noise)", color="green", linestyle="--")
ax.plot(X, y_pred, label="Fitted regression line", color="red")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()
st.pyplot(fig)

# ---------------- Dataset Preview ----------------
st.subheader("Data Preview")
df = pd.DataFrame({"X": X.flatten(), "y": y.flatten()})
st.dataframe(df.head())

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="synthetic_regression_data.csv", mime="text/csv")

st.caption("HW1 — Linear Regression demo with CRISP-DM process.")
