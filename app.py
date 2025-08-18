import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import zipfile
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# Load Data (from ZIP)
# ==============================
@st.cache_data
def load_data():
    zip_path = "household power consumption.zip"
    extracted_file = "household_power_consumption.txt"

    if not os.path.exists(extracted_file):
        if not os.path.exists(zip_path):
            st.error(f"{zip_path} not found.")
            return pd.DataFrame()
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

    if not os.path.exists(extracted_file):
        st.error(f"{extracted_file} not found inside ZIP.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            extracted_file,
            sep=";",
            parse_dates={"datetime": ["Date", "Time"]},
            infer_datetime_format=True,
            low_memory=False,
            na_values=["?"]
        )
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        st.error("Loaded CSV is empty.")
        return df

    df.set_index("datetime", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.asfreq("H").fillna(method="ffill")
    return df

# Load dataset
data = load_data()
if data.empty:
    st.stop()

# ==============================
# Load / Fast SARIMA
# ==============================
@st.cache_resource
def get_sarima_model(data, seasonal_order=(1,1,1,24), order=(1,1,1)):
    """Load SARIMA; if fails, fit on last 30 days of data silently."""
    try:
        model = SARIMAXResults.load("sarima_model.pkl")
        _ = model.get_forecast(steps=1)
        return model
    except:
        recent_series = data["Global_active_power"].iloc[-30*24:].astype(float)
        sarima_model = SARIMAX(recent_series, order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
        sarima_model_fit = sarima_model.fit(disp=False)
        sarima_model_fit.save("sarima_model.pkl")
        return sarima_model_fit

# ==============================
# Load Other Models
# ==============================
@st.cache_resource
def load_models():
    models = {}
    models["SARIMA"] = get_sarima_model(data)

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
# Forecasting Functions
# ==============================
def forecast_sarima(model, steps):
    try:
        forecast = model.get_forecast(steps=steps)
        return forecast.predicted_mean, forecast.conf_int()
    except Exception as e:
        st.error(f"SARIMA forecast error: {e}")
        return pd.Series(), pd.DataFrame()

def forecast_prophet(model, steps):
    try:
        future = model.make_future_dataframe(periods=steps, freq="H")
        forecast = model.predict(future)
        return forecast.tail(steps)[["ds", "yhat"]]
    except Exception as e:
        st.error(f"Prophet forecast error: {e}")
        return pd.DataFrame()

def forecast_xgb(model, steps, df):
    try:
        last_time = df.index[-1]
        future_idx = pd.date_range(start=last_time, periods=steps+1, freq="H")[1:]
        features = pd.DataFrame({
            "hour": future_idx.hour,
            "weekday": future_idx.weekday
        })
        preds = model.predict(features)
        return pd.Series(preds, index=future_idx)
    except Exception as e:
        st.error(f"XGBoost forecast error: {e}")
        return pd.Series()

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
            if preds.empty:
                st.stop()
            forecast_idx = pd.date_range(start=history.index[-1], periods=horizon+1, freq="H")[1:]
            forecast_series = pd.Series(preds.values, index=forecast_idx)

            mae = mean_absolute_error(test["Global_active_power"], forecast_series[:len(test)])
            rmse = np.sqrt(mean_squared_error(test["Global_active_power"], forecast_series[:len(test)]))

            plt.figure(figsize=(12,5))
            plt.plot(history.index, history["Global_active_power"], label="Actual")
            plt.plot(forecast_series.index, forecast_series, label="Forecast")
            plt.fill_between(forecast_series.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color="gray", alpha=0.3)
            plt.legend()
            st.pyplot(plt)

            st.success(f"SARIMA → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        elif selected_model == "Prophet" and "Prophet" in models:
            forecast = forecast_prophet(models["Prophet"], horizon)
            if forecast.empty:
                st.stop()
            forecast.set_index("ds", inplace=True)

            mae = mean_absolute_error(test["Global_active_power"], forecast["yhat"][:len(test)])
            rmse = np.sqrt(mean_squared_error(test["Global_active_power"], forecast["yhat"][:len(test)]))

            plt.figure(figsize=(12,5))
            plt.plot(history.index, history["Global_active_power"], label="Actual")
            plt.plot(forecast.index, forecast["yhat"], label="Forecast")
            plt.legend()
            st.pyplot(plt)

            st.success(f"Prophet → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        elif selected_model == "XGBoost" and "XGBoost" in models:
            preds = forecast_xgb(models["XGBoost"], horizon, data)
            if preds.empty:
                st.stop()

            mae = mean_absolute_error(test["Global_active_power"], preds[:len(test)])
            rmse = np.sqrt(mean_squared_error(test["Global_active_power"], preds[:len(test)]))

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
        sarima_preds = None
        if "SARIMA" in models:
            preds, _ = forecast_sarima(models["SARIMA"], horizon)
            if not preds.empty:
                forecast_idx = pd.date_range(start=history.index[-1], periods=horizon+1, freq="H")[1:]
                sarima_preds = pd.Series(preds.values, index=forecast_idx)
                results["SARIMA"] = {
                    "MAE": mean_absolute_error(test["Global_active_power"], sarima_preds[:len(test)]),
                    "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], sarima_preds[:len(test)]))
                }

        # Prophet
        forecast = None
        if "Prophet" in models:
            forecast = forecast_prophet(models["Prophet"], horizon)
            if not forecast.empty:
                forecast.set_index("ds", inplace=True)
                results["Prophet"] = {
                    "MAE": mean_absolute_error(test["Global_active_power"], forecast["yhat"][:len(test)]),
                    "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], forecast["yhat"][:len(test)]))
                }

        # XGBoost
        xgb_preds = None
        if "XGBoost" in models:
            xgb_preds = forecast_xgb(models["XGBoost"], horizon, data)
            if not xgb_preds.empty:
                results["XGBoost"] = {
                    "MAE": mean_absolute_error(test["Global_active_power"], xgb_preds[:len(test)]),
                    "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], xgb_preds[:len(test)]))
                }

        if not results:
            st.error("No models produced valid forecasts.")
            st.stop()

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
