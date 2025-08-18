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

    df = pd.read_csv(
        extracted_file,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        infer_datetime_format=True,
        low_memory=False,
        na_values=["?"]
    )

    df.set_index("datetime", inplace=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.asfreq("H").fillna(method="ffill")
    return df

data = load_data()
if data.empty:
    st.stop()

# ==============================
# SARIMA
# ==============================
@st.cache_resource
def get_sarima_model(data, seasonal_order=(1,1,1,24), order=(1,1,1)):
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
# Load Models
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
# Forecast Functions
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

    # Feature engineering
    hour = future_idx.hour
    dayofweek = future_idx.dayofweek
    is_weekend = (dayofweek >= 5).astype(int)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dayofweek / 7)

    lag_1 = df["Global_active_power"].shift(1).reindex(future_idx, method='ffill')
    lag_2 = df["Global_active_power"].shift(2).reindex(future_idx, method='ffill')
    lag_24 = df["Global_active_power"].shift(24).reindex(future_idx, method='ffill')

    features = pd.DataFrame({
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "lag_1": lag_1.values,
        "lag_2": lag_2.values,
        "lag_24": lag_24.values
    }, index=future_idx)

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

            mae = mean_absolute_error(test["Global_active_power"], preds[:len(test)])
            rmse = np.sqrt(mean_squared_error(test["Global_active_power"], preds[:len(test)]))

            plt.figure(figsize=(12,5))
            plt.plot(history.index, history["Global_active_power"], label="Actual")
            plt.plot(preds.index, preds, label="Forecast")
            plt.legend()
            st.pyplot(plt)
            st.success(f"XGBoost → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

# ==============================
# Compare All Models
# ==============================
elif mode == "Compare All Models":
    if st.button("Run Comparison"):
        results = {}
        sarima_preds = None
        forecast = None
        xgb_preds = None

        if "SARIMA" in models:
            preds, _ = forecast_sarima(models["SARIMA"], horizon)
            forecast_idx = pd.date_range(start=history.index[-1], periods=horizon+1, freq="H")[1:]
            sarima_preds = pd.Series(preds.values, index=forecast_idx)
            results["SARIMA"] = {
                "MAE": mean_absolute_error(test["Global_active_power"], sarima_preds[:len(test)]),
                "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], sarima_preds[:len(test)]))
            }

        if "Prophet" in models:
            forecast = forecast_prophet(models["Prophet"], horizon)
            forecast.set_index("ds", inplace=True)
            results["Prophet"] = {
                "MAE": mean_absolute_error(test["Global_active_power"], forecast["yhat"][:len(test)]),
                "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], forecast["yhat"][:len(test)]))
            }

        if "XGBoost" in models:
            xgb_preds = forecast_xgb(models["XGBoost"], horizon, data)
            results["XGBoost"] = {
                "MAE": mean_absolute_error(test["Global_active_power"], xgb_preds[:len(test)]),
                "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], xgb_preds[:len(test)]))
            }

        results_df = pd.DataFrame(results).T
        def highlight_best(s):
            is_min = s == s.min()
            return ['background-color: lightgreen' if v else '' for v in is_min]
        styled_df = results_df.style.apply(highlight_best, subset=["RMSE"])
        st.subheader("📊 Model Comparison (Last Horizon)")
        st.dataframe(styled_df)

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
