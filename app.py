import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import zipfile
import os
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    models = {}
    try:
        models["SARIMA"] = SARIMAXResults.load("sarima_model.pkl")
    except:
        st.warning("SARIMA model not found.")
    try:
        models["Prophet"] = joblib.load("prophet_model.pkl")
    except:
        st.warning("Prophet model not found.")
    try:
        models["XGBoost"] = joblib.load("xgb_model.pkl")
    except:
        st.warning("XGBoost model not found.")
    return models

models = load_models()

# ==============================
# Load Data (from ZIP)
# ==============================
@st.cache_data
def load_data():
    zip_path = "household_power_consumption.zip"
    csv_name = "household_power_consumption.csv"

    # unzip only if CSV not already extracted
    if not os.path.exists(csv_name):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

    df = pd.read_csv(csv_name, parse_dates=["datetime"], index_col="datetime")
    df = df.asfreq("H").fillna(method="ffill")  # hourly resample
    return df

data = load_data()

# ==============================
# Forecasting Functions
# ==============================
def forecast_sarima(model, steps):
    forecast = model.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()

def forecast_prophet(model, steps):
    future = model.make_future_dataframe(periods=steps, freq="H")
    forecast = model.predict(future)
    return forecast.tail(steps)[["ds", "yhat"]]

def forecast_xgb(model, steps, df):
    last_time = df.index[-1]
    future_idx = pd.date_range(start=last_time, periods=steps+1, freq="H")[1:]
    features = pd.DataFrame({
        "hour": future_idx.hour,
        "weekday": future_idx.weekday
    })
    preds = model.predict(features)
    return pd.Series(preds, index=future_idx)

# ==============================
# Streamlit UI
# ==============================
st.title("⚡ Energy Consumption Forecasting App")
st.write("Forecast household energy usage using SARIMA, Prophet, and XGBoost.")

horizon = st.slider("Select Forecast Horizon (hours)", min_value=1, max_value=48, value=24)

mode = st.radio("Choose Mode", ["Single Model", "Compare All Models"])

history = data.iloc[-(horizon+48):]
test = data.iloc[-horizon:]

# ==============================
# Single Model Mode
# ==============================
if mode == "Single Model":
    selected_model = st.selectbox("Choose a model", ["SARIMA", "Prophet", "XGBoost"])

    if st.button("Run Forecast"):
        if selected_model == "SARIMA" and "SARIMA" in models:
            preds, conf_int = forecast_sarima(models["SARIMA"], horizon)
            forecast_idx = pd.date_range(start=history.index[-1], periods=horizon+1, freq="H")[1:]
            forecast_series = pd.Series(preds.values, index=forecast_idx)

            mae = mean_absolute_error(test["Global_active_power"], forecast_series[:len(test)])
            rmse = mean_squared_error(test["Global_active_power"], forecast_series[:len(test)], squared=False)

            plt.figure(figsize=(12,5))
            plt.plot(history.index, history["Global_active_power"], label="Actual")
            plt.plot(forecast_series.index, forecast_series, label="Forecast")
            plt.fill_between(forecast_series.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color="gray", alpha=0.3)
            plt.legend()
            st.pyplot(plt)

            st.success(f"SARIMA → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        elif selected_model == "Prophet" and "Prophet" in models:
            forecast = forecast_prophet(models["Prophet"], horizon)
            forecast.set_index("ds", inplace=True)

            mae = mean_absolute_error(test["Global_active_power"], forecast["yhat"][:len(test)])
            rmse = mean_squared_error(test["Global_active_power"], forecast["yhat"][:len(test)], squared=False)

            plt.figure(figsize=(12,5))
            plt.plot(history.index, history["Global_active_power"], label="Actual")
            plt.plot(forecast.index, forecast["yhat"], label="Forecast")
            plt.legend()
            st.pyplot(plt)

            st.success(f"Prophet → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        elif selected_model == "XGBoost" and "XGBoost" in models:
            preds = forecast_xgb(models["XGBoost"], horizon, data)

            mae = mean_absolute_error(test["Global_active_power"], preds[:len(test)])
            rmse = mean_squared_error(test["Global_active_power"], preds[:len(test)], squared=False)

            plt.figure(figsize=(12,5))
            plt.plot(history.index, history["Global_active_power"], label="Actual")
            plt.plot(preds.index, preds, label="Forecast")
            plt.legend()
            st.pyplot(plt)

            st.success(f"XGBoost → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        else:
            st.error("Selected model is not available.")

# ==============================
# Compare All Models
# ==============================
elif mode == "Compare All Models":
    if st.button("Run Comparison"):
        results = {}

        # SARIMA
        if "SARIMA" in models:
            preds, _ = forecast_sarima(models["SARIMA"], horizon)
            forecast_idx = pd.date_range(start=history.index[-1], periods=horizon+1, freq="H")[1:]
            sarima_preds = pd.Series(preds.values, index=forecast_idx)
            results["SARIMA"] = {
                "MAE": mean_absolute_error(test["Global_active_power"], sarima_preds[:len(test)]),
                "RMSE": mean_squared_error(test["Global_active_power"], sarima_preds[:len(test)], squared=False)
            }
        else:
            sarima_preds = None

        # Prophet
        if "Prophet" in models:
            forecast = forecast_prophet(models["Prophet"], horizon)
            forecast.set_index("ds", inplace=True)
            results["Prophet"] = {
                "MAE": mean_absolute_error(test["Global_active_power"], forecast["yhat"][:len(test)]),
                "RMSE": mean_squared_error(test["Global_active_power"], forecast["yhat"][:len(test)], squared=False)
            }
        else:
            forecast = None

        # XGBoost
        if "XGBoost" in models:
            xgb_preds = forecast_xgb(models["XGBoost"], horizon, data)
            results["XGBoost"] = {
                "MAE": mean_absolute_error(test["Global_active_power"], xgb_preds[:len(test)]),
                "RMSE": mean_squared_error(test["Global_active_power"], xgb_preds[:len(test)], squared=False)
            }
        else:
            xgb_preds = None

        # Convert to DataFrame
        results_df = pd.DataFrame(results).T

        # Highlight best model (lowest RMSE)
        def highlight_best(s):
            is_min = s == s.min()
            return ['background-color: lightgreen' if v else '' for v in is_min]

        styled_df = results_df.style.apply(highlight_best, subset=["RMSE"])

        st.subheader("📊 Model Comparison (Last Horizon)")
        st.dataframe(styled_df)

        # Plot Comparison
        plt.figure(figsize=(12,5))
        plt.plot(history.index, history["Global_active_power"], label="Actual", color="black")
        if sarima_preds is not None:
            plt.plot(sarima_preds.index, sarima_preds, label="SARIMA")
        if forecast is not None:
            plt.plot(forecast.index, forecast["yhat"], label="Prophet")
        if xgb_preds is not None:
            plt.plot(xgb_preds.index, xgb_preds, label="XGBoost")
        plt.legend()
        st.pyplot(plt)
