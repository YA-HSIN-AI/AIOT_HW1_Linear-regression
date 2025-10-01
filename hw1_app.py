# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Step 2: Title and description
st.title("Interactive Linear Regression App — Amazon (AMZN) Price")
st.write("""
This app downloads **Amazon (AMZN)** historical prices from Yahoo Finance,
then fits a simple **linear regression of Price vs. Time Index**.
Adjust the date range and resampling frequency to see how it affects the fit.
""")

# ---------------- Sidebar controls ----------------
st.sidebar.subheader("Data Settings")
ticker = st.sidebar.text_input("Ticker", value="AMZN")
default_end = date.today()
default_start = default_end - timedelta(days=365)  # past 1 year
start_date = st.sidebar.date_input("Start date", value=default_start, max_value=default_end - timedelta(days=1))
end_date = st.sidebar.date_input("End date", value=default_end, min_value=start_date + timedelta(days=1))
freq = st.sidebar.selectbox("Resample frequency", options=["D (daily)", "W (weekly)", "M (monthly)"], index=0)

# Map display freq to pandas rule
freq_rule = {"D (daily)": "D", "W (weekly)": "W", "M (monthly)": "M"}[freq]

st.sidebar.subheader("Model Parameters")
# 只保留 n（可下采樣後再截取前 n 筆），其餘 a/c 不再用，因為改為真實數據
n = st.sidebar.slider('Max number of points to use (after resample)', min_value=20, max_value=2000, value=300, step=10)

# ---------------- Fetch data ----------------
@st.cache_data(show_spinner=True)
def load_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df

df_raw = load_prices(ticker, start_date, end_date)

if df_raw.empty:
    st.error("No data returned. Try another date range or ticker.")
    st.stop()

# Use Close price and resample
df = df_raw[['Close']].resample(freq_rule).last().dropna().copy()
df = df.head(n)  # limit to n points for speed/clarity

# Prepare X (time index as 0..n-1) and y (Close)
df = df.reset_index().rename(columns={'index': 'Date'})
df['t_index'] = np.arange(len(df))
X = df[['t_index']].values
y = df['Close'].values.reshape(-1, 1)

# ---------------- Fit linear regression ----------------
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# ---------------- Metrics ----------------
mse = mean_squared_error(y, y_pred)
r_squared = model.score(X, y)

# ---------------- Display metrics ----------------
st.subheader("Model Performance Metrics")
st.write(f"Ticker: **{ticker}**")
st.write(f"Range: **{start_date} → {end_date}**, resample: **{freq_rule}**, used points: **{len(df)}**")
st.write(f"Mean Squared Error (MSE): **{mse:.2f}**")
st.write(f"R-squared: **{r_squared:.4f}**")

# ---------------- Visualization ----------------
st.subheader("Data Visualization")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df['t_index'], df['Close'], label='Actual Close')
ax.plot(df['t_index'], y_pred.flatten(), label='Linear Fit')
ax.set_xlabel("Time Index (after resample)")
ax.set_ylabel("Price")
ax.set_title(f"{ticker} Close vs Linear Fit")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# Optional: table preview & download
st.write("Preview of dataset:")
st.dataframe(df[['Date', 'Close']].head())

csv = df[['Date', 'Close']].to_csv(index=False).encode("utf-8")
st.download_button("Download CSV (Date, Close)", csv, file_name=f"{ticker}_prices.csv", mime="text/csv")

st.caption("Note: This is a didactic linear trend fit (Price vs Time Index). For real forecasting, use proper time-series models.")
