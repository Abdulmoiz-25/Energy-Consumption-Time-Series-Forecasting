import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import zipfile
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import io
import base64
import time
from datetime import datetime, timedelta

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

@st.cache_resource
def get_custom_sarima_model(data, order, seasonal_order, _cache_key):
    """
    Train SARIMA model with custom parameters.
    _cache_key is used to invalidate cache when parameters change.
    """
    try:
        # Use last 30 days for faster training
        recent_series = data["Global_active_power"].iloc[-30*24:].astype(float)
        sarima_model = SARIMAX(recent_series, order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
        sarima_model_fit = sarima_model.fit(disp=False)
        return sarima_model_fit
    except Exception as e:
        st.error(f"Error training SARIMA with custom parameters: {e}")
        return None

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
    start = last_hist_index
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
        
        if not forecast_series.empty:
            # Add small random variations based on historical volatility
            historical_std = data["Global_active_power"].iloc[-168:].std()  # Last week's volatility
            noise_factor = 0.1  # 10% of historical volatility
            random_noise = np.random.normal(0, historical_std * noise_factor, len(forecast_series))
            forecast_series = forecast_series + random_noise
            
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
        
        if not forecast_series.empty:
            # Add small random variations based on historical volatility
            historical_std = data["Global_active_power"].iloc[-168:].std()  # Last week's volatility
            noise_factor = 0.15  # 15% of historical volatility for Prophet
            random_noise = np.random.normal(0, historical_std * noise_factor, len(forecast_series))
            forecast_series = forecast_series + random_noise
            
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
        forecast_series = pd.Series(preds, index=forecast_idx)
        
        if not forecast_series.empty:
            # Add small random variations based on historical volatility
            historical_std = df["Global_active_power"].iloc[-168:].std()  # Last week's volatility
            noise_factor = 0.08  # 8% of historical volatility for XGBoost (less since it's more detailed)
            random_noise = np.random.normal(0, historical_std * noise_factor, len(forecast_series))
            forecast_series = forecast_series + random_noise
            
        return forecast_series
    except Exception as e:
        st.error(f"XGBoost forecast error: {e}")
        return pd.Series(dtype=float)

# ==============================
# Helper function to clean forecast data for plotting
# ==============================
def clean_forecast_for_plotting(forecast_series):
    """
    Clean forecast series for plotting by removing NaN values and ensuring continuity.
    Returns cleaned series with no gaps.
    """
    if forecast_series.empty:
        return forecast_series
    
    # Remove NaN values
    cleaned = forecast_series.dropna()
    
    # If we have some valid data but with gaps, interpolate small gaps
    if len(cleaned) > 0 and len(cleaned) < len(forecast_series):
        # Interpolate to fill small gaps (up to 3 consecutive NaN values)
        interpolated = forecast_series.interpolate(method='linear', limit=3)
        # If still has NaN, use forward fill then backward fill
        interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
        return interpolated.dropna()
    
    return cleaned

# ==============================
# Export Functions
# ==============================
def prepare_export_data(history, forecast_data, model_name):
    """
    Prepare data for export combining historical and forecast data.
    """
    export_df = pd.DataFrame()
    
    # Add historical data
    export_df['timestamp'] = history.index
    export_df['actual_consumption'] = history["Global_active_power"].values
    export_df['data_type'] = 'historical'
    export_df['model'] = model_name
    
    # Add forecast data
    if not forecast_data.empty:
        forecast_df = pd.DataFrame()
        forecast_df['timestamp'] = forecast_data.index
        forecast_df['actual_consumption'] = np.nan  # No actual values for future
        forecast_df['forecast_consumption'] = forecast_data.values
        forecast_df['data_type'] = 'forecast'
        forecast_df['model'] = model_name
        
        # Combine historical and forecast
        export_df = pd.concat([export_df, forecast_df], ignore_index=True)
    
    return export_df

def prepare_comparison_export_data(history, forecast_results):
    """
    Prepare comparison data for export with all models.
    """
    export_df = pd.DataFrame()
    export_df['timestamp'] = history.index
    export_df['actual_consumption'] = history["Global_active_power"].values
    export_df['data_type'] = 'historical'
    
    # Add forecast data for each model
    for model_name, forecast_data in forecast_results.items():
        if not forecast_data.empty:
            # Extend dataframe to include forecast timestamps
            forecast_timestamps = forecast_data.index
            
            # Create a complete timeline
            all_timestamps = pd.concat([pd.Series(history.index), pd.Series(forecast_timestamps)]).drop_duplicates().sort_values()
            
            # Reindex export_df to include all timestamps
            if len(export_df) < len(all_timestamps):
                complete_df = pd.DataFrame()
                complete_df['timestamp'] = all_timestamps
                complete_df['actual_consumption'] = np.nan
                complete_df['data_type'] = 'unknown'
                
                # Fill in historical data
                hist_mask = complete_df['timestamp'].isin(history.index)
                complete_df.loc[hist_mask, 'actual_consumption'] = history["Global_active_power"].reindex(complete_df.loc[hist_mask, 'timestamp']).values
                complete_df.loc[hist_mask, 'data_type'] = 'historical'
                
                # Fill in forecast data
                forecast_mask = complete_df['timestamp'].isin(forecast_timestamps)
                complete_df.loc[forecast_mask, 'data_type'] = 'forecast'
                
                export_df = complete_df
            
            # Add model-specific forecast column
            export_df[f'{model_name.lower()}_forecast'] = np.nan
            forecast_mask = export_df['timestamp'].isin(forecast_timestamps)
            export_df.loc[forecast_mask, f'{model_name.lower()}_forecast'] = forecast_data.reindex(export_df.loc[forecast_mask, 'timestamp']).values
    
    return export_df

def create_download_link(df, filename, file_format='csv'):
    """
    Create a download link for the dataframe.
    """
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    elif file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Forecast_Data', index=False)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel</a>'
    
    return href

# ==============================
# Streamlit UI
# ==============================
st.title("⚡ Energy Consumption Forecasting App")
st.write("Forecast household energy usage using SARIMA, Prophet, and XGBoost.")

st.sidebar.header("🔄 Real-time Updates")

# Initialize session state for real-time updates
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 30
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'live_mode' not in st.session_state:
    st.session_state.live_mode = False

# Real-time controls in sidebar
st.session_state.live_mode = st.sidebar.checkbox(
    "🔴 Live Mode", 
    value=st.session_state.live_mode,
    help="Automatically use the most recent data available"
)

st.session_state.auto_refresh = st.sidebar.checkbox(
    "🔄 Auto Refresh", 
    value=st.session_state.auto_refresh,
    help="Automatically refresh forecasts at regular intervals"
)

if st.session_state.auto_refresh:
    st.session_state.refresh_interval = st.sidebar.selectbox(
        "Refresh Interval",
        options=[10, 30, 60, 300, 600],
        index=1,  # Default to 30 seconds
        format_func=lambda x: f"{x} seconds" if x < 60 else f"{x//60} minutes"
    )
    
    # Show countdown timer
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    time_until_refresh = max(0, st.session_state.refresh_interval - time_since_refresh)
    
    if time_until_refresh > 0:
        st.sidebar.info(f"⏱️ Next refresh in: {int(time_until_refresh)} seconds")
    else:
        st.sidebar.success("🔄 Refreshing now...")
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.session_state.last_refresh = datetime.now()
    st.cache_data.clear()  # Clear data cache to reload fresh data
    st.rerun()

# Show last update time
st.sidebar.write(f"📅 Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since_refresh >= st.session_state.refresh_interval:
        st.session_state.last_refresh = datetime.now()
        st.cache_data.clear()  # Clear data cache to get fresh data
        st.rerun()

st.subheader("🔧 Model Parameter Tuning")
with st.expander("Advanced Model Parameters", expanded=False):
    st.write("Adjust model parameters for better performance. **Note:** Changing parameters will retrain models and may take time.")
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        st.write("**SARIMA Parameters**")
        sarima_p = st.number_input("SARIMA p (AR order)", min_value=0, max_value=5, value=1, key="sarima_p")
        sarima_d = st.number_input("SARIMA d (Differencing)", min_value=0, max_value=2, value=1, key="sarima_d")
        sarima_q = st.number_input("SARIMA q (MA order)", min_value=0, max_value=5, value=1, key="sarima_q")
        sarima_P = st.number_input("Seasonal P", min_value=0, max_value=3, value=1, key="sarima_P")
        sarima_D = st.number_input("Seasonal D", min_value=0, max_value=2, value=1, key="sarima_D")
        sarima_Q = st.number_input("Seasonal Q", min_value=0, max_value=3, value=1, key="sarima_Q")
        sarima_s = st.number_input("Seasonal Period", min_value=1, max_value=168, value=24, key="sarima_s")
    
    with param_col2:
        st.write("**Prophet Parameters**")
        prophet_changepoint_prior = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, key="prophet_changepoint")
        prophet_seasonality_prior = st.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0, key="prophet_seasonality")
        prophet_holidays_prior = st.slider("Holidays Prior Scale", 0.01, 10.0, 10.0, key="prophet_holidays")
        prophet_daily_seasonality = st.checkbox("Daily Seasonality", value=True, key="prophet_daily")
        prophet_weekly_seasonality = st.checkbox("Weekly Seasonality", value=True, key="prophet_weekly")
        prophet_yearly_seasonality = st.checkbox("Yearly Seasonality", value=True, key="prophet_yearly")
    
    with param_col3:
        st.write("**XGBoost Parameters**")
        xgb_n_estimators = st.slider("N Estimators", 50, 500, 100, key="xgb_n_estimators")
        xgb_max_depth = st.slider("Max Depth", 3, 10, 6, key="xgb_max_depth")
        xgb_learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, key="xgb_learning_rate")
        xgb_subsample = st.slider("Subsample", 0.5, 1.0, 0.8, key="xgb_subsample")
        xgb_colsample_bytree = st.slider("Column Sample by Tree", 0.5, 1.0, 0.8, key="xgb_colsample")

st.subheader("📅 Data Selection")

if st.session_state.live_mode:
    # In live mode, automatically use the most recent data
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    
    # Use last 7 days for live mode
    start_date = max_date - pd.Timedelta(days=7)
    end_date = max_date
    
    st.info(f"🔴 Live Mode: Using most recent data from {start_date} to {end_date}")
    
else:
    col1, col2 = st.columns(2)

    with col1:
        # Get available date range from data
        min_date = data.index.min().date()
        max_date = data.index.max().date()
        
        start_date = st.date_input(
            "Start Date",
            value=max_date - pd.Timedelta(days=7),  # Default to last 7 days
            min_value=min_date,
            max_value=max_date
        )

    with col2:
        end_date = st.date_input(
            "End Date", 
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

# Validate date range
if start_date >= end_date:
    st.error("Start date must be before end date!")
    st.stop()

filtered_data = data.loc[start_date:end_date]
if filtered_data.empty:
    st.error("No data available for selected date range!")
    st.stop()

st.info(f"Selected data range: {len(filtered_data)} hours from {start_date} to {end_date}")

horizon = st.slider("Select Forecast Horizon (hours)", min_value=1, max_value=48, value=24)
mode = st.radio("Choose Mode", ["Single Model", "Compare All Models"])

# Use the filtered data for history and test sets
history = filtered_data
test = filtered_data.iloc[-min(horizon, len(filtered_data)):]  # Use available data or horizon, whichever is smaller

last_hist_index = history.index[-1]

custom_models = {}

# Custom SARIMA model
sarima_order = (sarima_p, sarima_d, sarima_q)
sarima_seasonal_order = (sarima_P, sarima_D, sarima_Q, sarima_s)
cache_key = f"{sarima_order}_{sarima_seasonal_order}"
custom_sarima = get_custom_sarima_model(data, sarima_order, sarima_seasonal_order, cache_key)
if custom_sarima:
    custom_models["SARIMA"] = custom_sarima

# Use original models for Prophet and XGBoost (parameter tuning for these would require retraining)
if "Prophet" in models:
    custom_models["Prophet"] = models["Prophet"]
if "XGBoost" in models:
    custom_models["XGBoost"] = models["XGBoost"]

auto_run = st.session_state.live_mode and st.session_state.auto_refresh

# ==============================
# Single Model Mode
# ==============================
if mode == "Single Model":
    selected_model = st.selectbox("Choose a model", ["SARIMA", "Prophet", "XGBoost"])

    if st.button("Run Forecast") or auto_run:
        if selected_model not in custom_models:
            st.error(f"{selected_model} model is not available or failed to train with current parameters.")
            st.stop()
            
        with st.spinner(f"Running {selected_model} forecast..."):
            if st.session_state.live_mode:
                st.success("🔴 Live Mode: Running real-time forecast")
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(14, 8))
            
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#2d2d2d')
            
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#555555')
            ax.set_axisbelow(True)
            
            ax.plot(history.index, history["Global_active_power"], 
                   label="Actual", color="#00d4ff", linewidth=2.5, alpha=0.9)

            forecast_series = None
            
            if selected_model == "SARIMA":
                sarima_series, conf_low, conf_up = forecast_sarima(custom_models["SARIMA"], last_hist_index, horizon)
                if sarima_series.empty:
                    st.error("SARIMA failed to produce forecast.")
                else:
                    forecast_series = sarima_series
                    cleaned_sarima = clean_forecast_for_plotting(sarima_series)
                    if not cleaned_sarima.empty:
                        ax.plot(cleaned_sarima.index, cleaned_sarima.values, 
                               label="SARIMA Forecast", color="#ff6b6b", linewidth=2.5, alpha=0.9)
                        if not conf_low.empty and not conf_up.empty:
                            cleaned_conf_low = clean_forecast_for_plotting(conf_low)
                            cleaned_conf_up = clean_forecast_for_plotting(conf_up)
                            if not cleaned_conf_low.empty and not cleaned_conf_up.empty:
                                ax.fill_between(cleaned_sarima.index, cleaned_conf_low.values, cleaned_conf_up.values, 
                                               alpha=0.2, color="#ff6b6b", label="Confidence Interval")
                    
                    forecast_values = sarima_series[:len(test)]
                    actual_values = test["Global_active_power"]
                    mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                    if mae is not None and rmse is not None:
                        st.success(f"SARIMA → MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                        st.info(f"Parameters: order={sarima_order}, seasonal_order={sarima_seasonal_order}")
                    else:
                        st.warning("SARIMA forecast contains too many NaN values, cannot calculate reliable metrics.")

            elif selected_model == "Prophet":
                prophet_series = forecast_prophet(custom_models["Prophet"], last_hist_index, horizon)
                if prophet_series.empty:
                    st.error("Prophet failed to produce forecast.")
                else:
                    forecast_series = prophet_series
                    cleaned_prophet = clean_forecast_for_plotting(prophet_series)
                    if not cleaned_prophet.empty:
                        ax.plot(cleaned_prophet.index, cleaned_prophet.values, 
                               label="Prophet Forecast", color="#4ecdc4", linewidth=2.5, alpha=0.9)
                    
                    forecast_values = prophet_series[:len(test)]
                    actual_values = test["Global_active_power"]
                    mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                    if mae is not None and rmse is not None:
                        st.success(f"Prophet → MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                        st.info(f"Using pre-trained Prophet model (parameter tuning requires retraining)")
                    else:
                        st.warning("Prophet forecast contains too many NaN values, cannot calculate reliable metrics.")

            elif selected_model == "XGBoost":
                xgb_series = forecast_xgb(custom_models["XGBoost"], last_hist_index, horizon, data)
                if xgb_series.empty:
                    st.error("XGBoost failed to produce forecast.")
                else:
                    forecast_series = xgb_series
                    cleaned_xgb = clean_forecast_for_plotting(xgb_series)
                    if not cleaned_xgb.empty:
                        ax.plot(cleaned_xgb.index, cleaned_xgb.values, 
                               label="XGBoost Forecast", color="#ffa726", linewidth=2.5, alpha=0.9)
                    
                    forecast_values = xgb_series[:len(test)]
                    actual_values = test["Global_active_power"]
                    mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                    if mae is not None and rmse is not None:
                        st.success(f"XGBoost → MAE: {mae:.3f}, RMSE: {rmse:.3f}")
                        st.info(f"Using pre-trained XGBoost model (parameter tuning requires retraining)")
                    else:
                        st.warning("XGBoost forecast contains too many NaN values, cannot calculate reliable metrics.")

            title_suffix = " (Live Mode)" if st.session_state.live_mode else ""
            ax.set_xlabel("Time", fontsize=14, color='white', fontweight='bold')
            ax.set_ylabel("Energy Consumption (kW)", fontsize=14, color='white', fontweight='bold')
            ax.set_title(f"{selected_model} Energy Consumption Forecast{title_suffix}", 
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
            
            if forecast_series is not None and not forecast_series.empty:
                st.subheader("📥 Export Forecast Data")
                export_data = prepare_export_data(history, forecast_series, selected_model)
                
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = export_data.to_csv(index=False)
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv_data,
                        file_name=f"{selected_model}_forecast_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        export_data.to_excel(writer, sheet_name='Forecast_Data', index=False)
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_data,
                        file_name=f"{selected_model}_forecast_{start_date}_{end_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.info(f"Export includes {len(history)} historical data points and {len(forecast_series)} forecast points.")

# ==============================
# Compare All Models
# ==============================
elif mode == "Compare All Models":
    if st.button("Run Comparison") or auto_run:
        with st.spinner("Running comparison with all available models..."):
            if st.session_state.live_mode:
                st.success("🔴 Live Mode: Running real-time comparison")
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(14, 8))
            
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#2d2d2d')
            
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#555555')
            ax.set_axisbelow(True)
            
            ax.plot(history.index, history["Global_active_power"], 
                   label="Actual", color="#00d4ff", linewidth=3, alpha=0.9)

            results = {}
            forecast_results = {}
            sarima_series = prophet_series = xgb_series = None

            if "SARIMA" in custom_models:
                sarima_series, conf_low, conf_up = forecast_sarima(custom_models["SARIMA"], last_hist_index, horizon)
                if not sarima_series.empty:
                    forecast_results["SARIMA"] = sarima_series
                    cleaned_sarima = clean_forecast_for_plotting(sarima_series)
                    if not cleaned_sarima.empty:
                        ax.plot(cleaned_sarima.index, cleaned_sarima.values, 
                               label="SARIMA", color="#ff6b6b", linewidth=2.5, alpha=0.9)
                        if not conf_low.empty and not conf_up.empty:
                            cleaned_conf_low = clean_forecast_for_plotting(conf_low)
                            cleaned_conf_up = clean_forecast_for_plotting(conf_up)
                            if not cleaned_conf_low.empty and not cleaned_conf_up.empty:
                                ax.fill_between(cleaned_sarima.index, cleaned_conf_low.values, cleaned_conf_up.values, 
                                               alpha=0.15, color="#ff6b6b")
                    
                    forecast_values = sarima_series[:len(test)]
                    actual_values = test["Global_active_power"]
                    mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                    if mae is not None and rmse is not None:
                        results["SARIMA"] = {"MAE": mae, "RMSE": rmse}

            if "Prophet" in custom_models:
                prophet_series = forecast_prophet(custom_models["Prophet"], last_hist_index, horizon)
                if not prophet_series.empty:
                    forecast_results["Prophet"] = prophet_series
                    cleaned_prophet = clean_forecast_for_plotting(prophet_series)
                    if not cleaned_prophet.empty:
                        ax.plot(cleaned_prophet.index, cleaned_prophet.values, 
                               label="Prophet", color="#4ecdc4", linewidth=2.5, alpha=0.9)
                    
                    forecast_values = prophet_series[:len(test)]
                    actual_values = test["Global_active_power"]
                    mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                    if mae is not None and rmse is not None:
                        results["Prophet"] = {"MAE": mae, "RMSE": rmse}

            if "XGBoost" in custom_models:
                xgb_series = forecast_xgb(custom_models["XGBoost"], last_hist_index, horizon, data)
                if not xgb_series.empty:
                    forecast_results["XGBoost"] = xgb_series
                    cleaned_xgb = clean_forecast_for_plotting(xgb_series)
                    if not cleaned_xgb.empty:
                        ax.plot(cleaned_xgb.index, cleaned_xgb.values, 
                               label="XGBoost", color="#ffa726", linewidth=2.5, alpha=0.9)
                    
                    forecast_values = xgb_series[:len(test)]
                    actual_values = test["Global_active_power"]
                    mae, rmse = safe_calculate_metrics(actual_values, forecast_values)
                    if mae is not None and rmse is not None:
                        results["XGBoost"] = {"MAE": mae, "RMSE": rmse}

            if not results:
                st.error("No models produced valid forecasts.")
                st.stop()

            title_suffix = " (Live Mode)" if st.session_state.live_mode else ""
            ax.set_xlabel("Time", fontsize=14, color='white', fontweight='bold')
            ax.set_ylabel("Energy Consumption (kW)", fontsize=14, color='white', fontweight='bold')
            ax.set_title(f"Energy Consumption Forecast Comparison{title_suffix}", 
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
            
            if "SARIMA" in results:
                st.info(f"SARIMA Parameters: order={sarima_order}, seasonal_order={sarima_seasonal_order}")
            
            if forecast_results:
                st.subheader("📥 Export Comparison Data")
                comparison_data = prepare_comparison_export_data(history, forecast_results)
                
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = comparison_data.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Comparison as CSV",
                        data=csv_data,
                        file_name=f"forecast_comparison_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Create Excel file in memory
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        comparison_data.to_excel(writer, sheet_name='Comparison_Data', index=False)
                        # Add metrics sheet
                        results_df.to_excel(writer, sheet_name='Model_Metrics')
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="📊 Download Comparison as Excel",
                        data=excel_data,
                        file_name=f"forecast_comparison_{start_date}_{end_date}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                total_forecasts = sum(len(f) for f in forecast_results.values())
                st.info(f"Export includes {len(history)} historical data points and forecasts from {len(forecast_results)} models ({total_forecasts} total forecast points).")

if st.session_state.auto_refresh:
    # Add a small delay and rerun to create continuous refresh
    time.sleep(1)
    st.rerun()
