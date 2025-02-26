#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import fastapi
import uvicorn
from fastapi import FastAPI
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

# FastAPI Backend Setup
app = FastAPI()

@app.get("/forecast")
def get_forecast():
    return {"message": "API for Renewable Energy Forecasting"}

@app.post("/predict")
def predict_energy(data: dict):
    df = pd.DataFrame(data)
    model_prophet = Prophet()
    model_prophet.fit(df)
    future = model_prophet.make_future_dataframe(periods=30)
    forecast = model_prophet.predict(future)
    return forecast[['ds', 'yhat']].to_dict(orient='records')

# Title of the App
st.title("Renewable Energy Forecasting & Grid Optimization Suite")
st.subheader("A Full-Stack Software Tool for Energy Forecasting & Grid Optimization")

# Sidebar options
st.sidebar.header("Upload Energy Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview")
    st.write(df.head())

    # Ensure the dataset is not empty before proceeding
    if not df.empty:
    
    # Data preprocessing
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['energy_generated']
    df = df[['ds', 'y']]

    # Prophet Forecasting Model
    model_prophet = Prophet()
    model_prophet.fit(df)
    future = model_prophet.make_future_dataframe(periods=30)
    forecast = model_prophet.predict(future)
    
    st.write("### Renewable Energy Forecast (Prophet)")
    fig = px.line(forecast, x='ds', y='yhat', title='Energy Generation Forecast (Prophet)')
    st.plotly_chart(fig)

    # LSTM Deep Learning Model for Energy Forecasting
    st.write("### Deep Learning Forecasting Model (LSTM)")
    data = df['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(10, len(data_scaled)):
        X.append(data_scaled[i-10:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model_lstm = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X, y, epochs=10, batch_size=1, verbose=2)

    predictions = model_lstm.predict(X)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    df['LSTM Forecast'] = np.nan
    df.iloc[-len(predictions):, df.columns.get_loc('LSTM Forecast')] = predictions.flatten()

    fig = px.line(df, x='ds', y=['y', 'LSTM Forecast'], title='Energy Generation Forecast (LSTM)')
    st.plotly_chart(fig)

# REST API Integration using SerpAPI
st.write("## API Connectivity for Renewable Energy Systems")
st.write("Fetching real-time weather data using SerpAPI...")

# Fetch SerpAPI Key from Streamlit Secrets
SERP_API_KEY = st.secrets("SERPAPI_KEY")  # Ensure your key is stored in Streamlit Secrets
CITY = "Oldenburg"
serpapi_url = f"https://serpapi.com/search.json?q=weather+{CITY}&hl=en&api_key={SERP_API_KEY}"

try:
    response = requests.get(serpapi_url)
    data = response.json()

    if "weather" in data:
        temp = data["weather"]["temperature"]
        condition = data["weather"]["description"]
        humidity = data["weather"]["humidity"]
        wind_speed = data["weather"]["wind"]["speed"]

        st.write(f"🌡 Temperature: {temp}")
        st.write(f"☁ Condition: {condition}")
        st.write(f"💧 Humidity: {humidity}%")
        st.write(f"💨 Wind Speed: {wind_speed} km/h")
    else:
        st.write("❌ Failed to fetch weather data from SerpAPI.")
except Exception as e:
    st.write(f"❌ Error fetching weather data: {e}")

# Energy Grid Load Balancing
st.write("## Grid Load Balancing & Energy Management")
load_data = np.random.randint(50, 150, size=30)
energy_data = np.random.randint(60, 140, size=30)

grid_df = pd.DataFrame({
    'Day': range(1, 31),
    'Grid Load (MW)': load_data,
    'Energy Supplied (MW)': energy_data
})

fig = px.line(grid_df, x='Day', y=['Grid Load (MW)', 'Energy Supplied (MW)'], title='Grid Load vs Energy Supplied')
st.plotly_chart(fig)

# Virtual Power Plant (VPP) Simulation
st.write("## Virtual Power Plant Simulation")
st.write("Simulating decentralized energy sources...")
vpp_data = np.random.randint(10, 50, size=10)

vpp_df = pd.DataFrame({
    'Plant ID': range(1, 11),
    'Energy Output (MW)': vpp_data
})
fig = px.bar(vpp_df, x='Plant ID', y='Energy Output (MW)', title='Virtual Power Plant Energy Output')
st.plotly_chart(fig)

# Predictive Maintenance for Renewable Assets
st.write("## Predictive Maintenance for Wind Turbines & Solar Panels")
st.write("Monitoring asset performance and failure risks...")
maintenance_data = np.random.choice(['Operational', 'Maintenance Required', 'Critical'], size=5)
wind_turbine_df = pd.DataFrame({
    'Turbine ID': range(1, 6),
    'Status': maintenance_data
})
st.write(wind_turbine_df)

# Start FastAPI Server in Background
# if __name__ == "__main__":
# uvicorn.run(app, host="0.0.0.0", port=8000)

