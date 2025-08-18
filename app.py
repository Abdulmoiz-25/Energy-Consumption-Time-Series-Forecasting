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
def safe_calculate_metrics(actual_values, forecast_values):
    """
    Calculate metrics while handling NaN values by filtering them out.
    Returns MAE and RMSE, or None if insufficient valid data.
    """
    # Convert to numpy arrays and find valid (non-NaN) indices
    actual_array = np.array(actual_values)
    forecast_array = np.array(forecast_values)
    
    # Find indices where both actual and forecast are not NaN
    valid_mask = ~(np.isnan(actual_array) | np.isnan(forecast_array))
    
    if np.sum(valid_mask) < 2:  # Need at least 2 valid points
        return None, None
    
    # Filter to only valid data points
    valid_actual = actual_array[valid_mask]
    valid_forecast = forecast_array[valid_mask]
    
    try:
        mae = mean_absolute_error(valid_actual, valid_forecast)
        rmse = np.sqrt(mean_squared_error(valid_actual, valid_forecast))
        return mae, rmse
    except Exception:
        return None, None

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
        # Create future dataframe with proper alignment
        future = model.make_future_dataframe(periods=steps, freq="H")
        forecast_df = model.predict(future)
        
        # Get the last `steps` predictions (the future predictions)
        future_predictions = forecast_df.tail(steps).copy()
        
        # Instead of reindexing which can create NaN, directly map the values
        # The Prophet predictions should be in chronological order
        if len(future_predictions) == steps:
            # Direct mapping of values to our forecast index
            forecast_values = future_predictions["yhat"].values
            forecast_series = pd.Series(forecast_values, index=forecast_idx)
        else:
            # Fallback: interpolate if lengths don't match
            future_predictions.set_index("ds", inplace=True)
            reindexed = future_predictions["yhat"].reindex(forecast_idx, method='nearest')
            # Fill any remaining NaN values with the last valid prediction
            if reindexed.isna().any():
                last_valid = future_predictions["yhat"].iloc[-1]
                reindexed = reindexed.fillna(last_valid)
            forecast_series = reindexed
        
        return forecast_series
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
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#2d2d2d')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#555555')
        ax.set_axisbelow(True)
        
        ax.plot(history.index, history["Global_active_power"], 
               label="Actual", color="#00d4ff", linewidth=2.5, alpha=0.9)

        if selected_model == "SARIMA" and "SARIMA" in models:
            sarima_series, conf_low, conf_up = forecast_sarima(models["SARIMA"], last_hist_index, horizon)
            if sarima_series.empty:
                st.error("SARIMA failed to produce forecast.")
            else:
                forecast_values = sarima_series[:len(test)]
                actual_values = test["Global_active_power"]
                mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                if mae is not None and rmse is not None:
                    st.success(f"SARIMA → MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                else:
                    st.warning("SARIMA forecast contains too many NaN values, cannot calculate reliable metrics.")
                ax.plot(sarima_series.index, sarima_series.values, 
                       label="SARIMA Forecast", color="#ff6b6b", linewidth=2.5, alpha=0.9)
                if not conf_low.empty and not conf_up.empty:
                    ax.fill_between(sarima_series.index, conf_low.values, conf_up.values, 
                                   alpha=0.2, color="#ff6b6b", label="Confidence Interval")

        elif selected_model == "Prophet" and "Prophet" in models:
            prophet_series = forecast_prophet(models["Prophet"], last_hist_index, horizon)
            if prophet_series.empty:
                st.error("Prophet failed to produce forecast.")
            else:
                forecast_values = prophet_series[:len(test)]
                actual_values = test["Global_active_power"]
                mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                if mae is not None and rmse is not None:
                    st.success(f"Prophet → MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                else:
                    st.warning("Prophet forecast contains too many NaN values, cannot calculate reliable metrics.")
                ax.plot(prophet_series.index, prophet_series.values, 
                       label="Prophet Forecast", color="#4ecdc4", linewidth=2.5, alpha=0.9)

        elif selected_model == "XGBoost" and "XGBoost" in models:
            xgb_series = forecast_xgb(models["XGBoost"], last_hist_index, horizon, data)
            if xgb_series.empty:
                st.error("XGBoost failed to produce forecast.")
            else:
                forecast_values = xgb_series[:len(test)]
                actual_values = test["Global_active_power"]
                mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                if mae is not None and rmse is not None:
                    st.success(f"XGBoost → MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                else:
                    st.warning("XGBoost forecast contains too many NaN values, cannot calculate reliable metrics.")
                ax.plot(xgb_series.index, xgb_series.values, 
                       label="XGBoost Forecast", color="#ffa726", linewidth=2.5, alpha=0.9)

        else:
            st.error("Selected model is not available.")

        ax.set_xlabel("Time", fontsize=14, color='white', fontweight='bold')
        ax.set_ylabel("Energy Consumption (kW)", fontsize=14, color='white', fontweight='bold')
        ax.set_title(f"{selected_model} Energy Consumption Forecast", 
                    fontsize=18, color='white', fontweight='bold', pad=20)
        
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                          fontsize=12, facecolor='#3d3d3d', edgecolor='#555555')
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color('white')
        
        ax.tick_params(axis='both', which='major', labelsize=11, colors='white')
        ax.spines['bottom'].set_color('#555555')
        ax.spines['top'].set_color('#555555')
        ax.spines['right'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        
        plt.tight_layout()
        st.pyplot(fig)

# ==============================
# Compare All Models
# ==============================
elif mode == "Compare All Models":
    if st.button("Run Comparison"):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#2d2d2d')
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#555555')
        ax.set_axisbelow(True)
        
        ax.plot(history.index, history["Global_active_power"], 
               label="Actual", color="#00d4ff", linewidth=3, alpha=0.9)

        results = {}
        sarima_series = prophet_series = xgb_series = None

        if "SARIMA" in models:
            sarima_series, conf_low, conf_up = forecast_sarima(models["SARIMA"], last_hist_index, horizon)
            if not sarima_series.empty:
                ax.plot(sarima_series.index, sarima_series.values, 
                       label="SARIMA", color="#ff6b6b", linewidth=2.5, alpha=0.9)
                if not conf_low.empty and not conf_up.empty:
                    ax.fill_between(sarima_series.index, conf_low.values, conf_up.values, 
                                   alpha=0.15, color="#ff6b6b")
                forecast_values = sarima_series[:len(test)]
                actual_values = test["Global_active_power"]
                mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                if mae is not None and rmse is not None:
                    results["SARIMA"] = {"MAE": mae, "RMSE": rmse}

        if "Prophet" in models:
            prophet_series = forecast_prophet(models["Prophet"], last_hist_index, horizon)
            if not prophet_series.empty:
                ax.plot(prophet_series.index, prophet_series.values, 
                       label="Prophet", color="#4ecdc4", linewidth=2.5, alpha=0.9)
                forecast_values = prophet_series[:len(test)]
                actual_values = test["Global_active_power"]
                mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                if mae is not None and rmse is not None:
                    results["Prophet"] = {"MAE": mae, "RMSE": rmse}

        if "XGBoost" in models:
            xgb_series = forecast_xgb(models["XGBoost"], last_hist_index, horizon, data)
            if not xgb_series.empty:
                ax.plot(xgb_series.index, xgb_series.values, 
                       label="XGBoost", color="#ffa726", linewidth=2.5, alpha=0.9)
                forecast_values = xgb_series[:len(test)]
                actual_values = test["Global_active_power"]
                mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                if mae is not None and rmse is not None:
                    results["XGBoost"] = {"MAE": mae, "RMSE": rmse}

        if not results:
            st.error("No models produced valid forecasts.")
            st.stop()

        ax.set_xlabel("Time", fontsize=14, color='white', fontweight='bold')
        ax.set_ylabel("Energy Consumption (kW)", fontsize=14, color='white', fontweight='bold')
        ax.set_title("Energy Consumption Forecast Comparison", 
                    fontsize=18, color='white', fontweight='bold', pad=20)
        
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                          fontsize=12, facecolor='#3d3d3d', edgecolor='#555555')
        legend.get_frame().set_alpha(0.9)
        for text in legend.get_texts():
            text.set_color('white')
        
        ax.tick_params(axis='both', which='major', labelsize=11, colors='white')
        ax.spines['bottom'].set_color('#555555')
        ax.spines['top'].set_color('#555555')
        ax.spines['right'].set_color('#555555')
        ax.spines['left'].set_color('#555555')
        
        plt.tight_layout()
        st.pyplot(fig)

        results_df = pd.DataFrame(results).T
        def highlight_best(s):
            is_min = s == s.min()
            return ['background-color: lightgreen' if v else '' for v in is_min]
        styled_df = results_df.style.apply(highlight_best, subset=["RMSE"])
        st.subheader("📊 Model Comparison (Last Horizon)")
        st.dataframe(styled_df)
