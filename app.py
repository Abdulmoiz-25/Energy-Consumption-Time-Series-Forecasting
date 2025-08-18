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
    # Ensure strictly hourly index and forward-fill
    df = df.asfreq("H").fillna(method="ffill")
    return df

data = load_data()
if data.empty:
    st.stop()

# ==============================
# SARIMA (fast fit on last 30 days if needed)
# ==============================
@st.cache_resource
def get_sarima_model(data, seasonal_order=(1,1,1,24), order=(1,1,1)):
    try:
        model = SARIMAXResults.load("sarima_model.pkl")
        # quick test
        _ = model.get_forecast(steps=1)
        return model
    except:
        # silent refit using last 30 days (~720 hours) for speed
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
# Forecast Functions (return series aligned to forecast_idx)
# ==============================
def make_forecast_index(last_hist_index, horizon):
    # start forecast 1 hour after last history timestamp
    start = last_hist_index + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=horizon, freq="H")

def forecast_sarima(model, last_hist_index, steps):
    """
    Returns: forecast_series, conf_lower, conf_upper
    Each as pd.Series aligned to forecast_idx
    """
    forecast_idx = make_forecast_index(last_hist_index, steps)
    try:
        forecast_obj = model.get_forecast(steps=steps)
        preds = np.asarray(forecast_obj.predicted_mean)
        conf = forecast_obj.conf_int()
        # Safely extract conf bounds, handle shorter conf results
        if conf is not None and len(conf) >= steps:
            lower = np.asarray(conf.iloc[:steps, 0])
            upper = np.asarray(conf.iloc[:steps, 1])
        else:
            lower = np.full(steps, np.nan)
            upper = np.full(steps, np.nan)

        forecast_series = pd.Series(preds[:steps], index=forecast_idx)
        conf_lower = pd.Series(lower, index=forecast_idx)
        conf_upper = pd.Series(upper, index=forecast_idx)
        return forecast_series, conf_lower, conf_upper
    except Exception as e:
        st.error(f"SARIMA forecast error: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

def forecast_prophet(model, last_hist_index, steps):
    """
    Returns: forecast_series aligned to forecast_idx
    """
    forecast_idx = make_forecast_index(last_hist_index, steps)
    try:
        future = model.make_future_dataframe(periods=steps, freq="H")
        forecast_df = model.predict(future)
        # take last `steps` rows and set index to 'ds'
        last_forecast = forecast_df.tail(steps)[["ds", "yhat"]].copy()
        last_forecast.set_index("ds", inplace=True)
        # Reindex to our forecast_idx to ensure alignment (interpolate if needed)
        last_forecast = last_forecast.reindex(forecast_idx)
        # If some values missing, forward/backfill
        last_forecast["yhat"].fillna(method="ffill", inplace=True)
        last_forecast["yhat"].fillna(method="bfill", inplace=True)
        return pd.Series(last_forecast["yhat"].values, index=forecast_idx)
    except Exception as e:
        st.error(f"Prophet forecast error: {e}")
        return pd.Series(dtype=float)

def forecast_xgb(model, last_hist_index, steps, df):
    """
    Recreate feature engineering used during training and predict.
    Returns pd.Series aligned to forecast_idx
    """
    forecast_idx = make_forecast_index(last_hist_index, steps)
    try:
        hour = forecast_idx.hour
        dayofweek = forecast_idx.dayofweek
        is_weekend = (dayofweek >= 5).astype(int)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * dayofweek / 7)

        # build lag features from historical series (most recent values)
        # We align the lags by taking historic values and reindexing to future
        hist = df["Global_active_power"]
        # produce a series with same index as forecast_idx by ffill from last values
        lag_1 = hist.shift(1).reindex(forecast_idx, method='ffill').values
        lag_2 = hist.shift(2).reindex(forecast_idx, method='ffill').values
        lag_24 = hist.shift(24).reindex(forecast_idx, method='ffill').values

        features = pd.DataFrame({
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_24": lag_24
        }, index=forecast_idx)

        preds = model.predict(features)
        return pd.Series(preds, index=forecast_idx)
    except Exception as e:
        st.error(f"XGBoost forecast error: {e}")
        return pd.Series(dtype=float)

# ==============================
# Streamlit UI
# ==============================
st.title("⚡ Energy Consumption Forecasting App")
st.write("Forecast household energy usage using SARIMA, Prophet, and XGBoost.")

horizon = st.slider("Select Forecast Horizon (hours)", min_value=1, max_value=48, value=24)
mode = st.radio("Choose Mode", ["Single Model", "Compare All Models"])

# Define history window to show: last (horizon + 48) hours
history_window = horizon + 48
history = data.iloc[-history_window:]
test = data.iloc[-horizon:]

last_hist_index = history.index[-1]

# ==============================
# Single Model Mode
# ==============================
if mode == "Single Model":
    selected_model = st.selectbox("Choose a model", ["SARIMA", "Prophet", "XGBoost"])

    if st.button("Run Forecast"):
        plt.figure(figsize=(12,5))

        # plot recent history
        plt.plot(history.index, history["Global_active_power"], label="Actual", color="black")

        if selected_model == "SARIMA" and "SARIMA" in models:
            sarima_series, conf_low, conf_up = forecast_sarima(models["SARIMA"], last_hist_index, horizon)
            if sarima_series.empty:
                st.error("SARIMA failed to produce forecast.")
            else:
                mae = mean_absolute_error(test["Global_active_power"], sarima_series[:len(test)])
                rmse = np.sqrt(mean_squared_error(test["Global_active_power"], sarima_series[:len(test)]))
                plt.plot(sarima_series.index, sarima_series.values, label="SARIMA")
                # plot conf interval if available
                if not conf_low.empty and not conf_up.empty:
                    plt.fill_between(sarima_series.index, conf_low.values, conf_up.values, alpha=0.25)
                st.success(f"SARIMA → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        elif selected_model == "Prophet" and "Prophet" in models:
            prophet_series = forecast_prophet(models["Prophet"], last_hist_index, horizon)
            if prophet_series.empty:
                st.error("Prophet failed to produce forecast.")
            else:
                mae = mean_absolute_error(test["Global_active_power"], prophet_series[:len(test)])
                rmse = np.sqrt(mean_squared_error(test["Global_active_power"], prophet_series[:len(test)]))
                plt.plot(prophet_series.index, prophet_series.values, label="Prophet")
                st.success(f"Prophet → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        elif selected_model == "XGBoost" and "XGBoost" in models:
            xgb_series = forecast_xgb(models["XGBoost"], last_hist_index, horizon, data)
            if xgb_series.empty:
                st.error("XGBoost failed to produce forecast.")
            else:
                mae = mean_absolute_error(test["Global_active_power"], xgb_series[:len(test)])
                rmse = np.sqrt(mean_squared_error(test["Global_active_power"], xgb_series[:len(test)]))
                plt.plot(xgb_series.index, xgb_series.values, label="XGBoost")
                st.success(f"XGBoost → MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        else:
            st.error("Selected model is not available.")

        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

# ==============================
# Compare All Models
# ==============================
elif mode == "Compare All Models":
    if st.button("Run Comparison"):
        plt.figure(figsize=(12,5))
        plt.plot(history.index, history["Global_active_power"], label="Actual", color="black")

        results = {}
        sarima_series = prophet_series = xgb_series = None

        # SARIMA
        if "SARIMA" in models:
            sarima_series, conf_low, conf_up = forecast_sarima(models["SARIMA"], last_hist_index, horizon)
            if not sarima_series.empty:
                plt.plot(sarima_series.index, sarima_series.values, label="SARIMA")
                if not conf_low.empty and not conf_up.empty:
                    plt.fill_between(sarima_series.index, conf_low.values, conf_up.values, alpha=0.15)
                results["SARIMA"] = {
                    "MAE": mean_absolute_error(test["Global_active_power"], sarima_series[:len(test)]),
                    "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], sarima_series[:len(test)]))
                }

        # Prophet
        if "Prophet" in models:
            prophet_series = forecast_prophet(models["Prophet"], last_hist_index, horizon)
            if not prophet_series.empty:
                plt.plot(prophet_series.index, prophet_series.values, label="Prophet")
                results["Prophet"] = {
                    "MAE": mean_absolute_error(test["Global_active_power"], prophet_series[:len(test)]),
                    "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], prophet_series[:len(test)]))
                }

        # XGBoost
        if "XGBoost" in models:
            xgb_series = forecast_xgb(models["XGBoost"], last_hist_index, horizon, data)
            if not xgb_series.empty:
                plt.plot(xgb_series.index, xgb_series.values, label="XGBoost")
                results["XGBoost"] = {
                    "MAE": mean_absolute_error(test["Global_active_power"], xgb_series[:len(test)]),
                    "RMSE": np.sqrt(mean_squared_error(test["Global_active_power"], xgb_series[:len(test)]))
                }

        if not results:
            st.error("No models produced valid forecasts.")
            st.stop()

        plt.legend()
        plt.tight_layout()
        st.pyplot(plt)

        # Show comparison table
        results_df = pd.DataFrame(results).T
        def highlight_best(s):
            is_min = s == s.min()
            return ['background-color: lightgreen' if v else '' for v in is_min]
        styled_df = results_df.style.apply(highlight_best, subset=["RMSE"])
        st.subheader("📊 Model Comparison (Last Horizon)")
        st.dataframe(styled_df)
