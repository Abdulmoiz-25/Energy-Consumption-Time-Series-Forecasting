# 📘 DeveloperHub Task 8 – Energy Consumption Forecasting App

## 📌 Project Objective  
Forecast household energy consumption using **advanced time series models** to predict future energy usage patterns. The interactive dashboard utilizes **SARIMA, Prophet, and XGBoost** models for accurate forecasting with comprehensive visualization and performance metrics.

---

## 📁 Dataset  
- **Name**: Household Power Consumption Dataset  
- **Source**: UCI Machine Learning Repository  
- **Features include**:  
  - Global Active Power (household global minute-averaged active power in kilowatts)
  - Date and Time (minute-level timestamps)
  - Various electrical measurements and sub-metering data
- **Frequency**: Converted to hourly data for forecasting

---

## 🛠️ Tools & Libraries Used  
- **Streamlit** – interactive web dashboard deployment  
- **Pandas** – data manipulation and time series handling  
- **NumPy** – numerical operations and array processing  
- **Statsmodels** – SARIMA time series modeling  
- **Prophet** – Facebook's time series forecasting tool  
- **XGBoost** – gradient boosting for machine learning forecasting  
- **Scikit-learn** – model evaluation metrics (MAE, RMSE)  
- **Matplotlib** – advanced plotting with dark theme styling  
- **Joblib** – model serialization and caching  

---

## 🚀 Approach  

### 🔍 1. Data Loading & Preprocessing  
- Automatic ZIP file extraction and CSV parsing  
- Date/time combination and datetime indexing  
- Hourly frequency resampling with forward-fill interpolation  
- Missing value handling and data type conversion  
- Cached data loading for improved performance  

### 🤖 2. Multi-Model Forecasting  
- **SARIMA**: Statistical time series model with seasonal components  
- **Prophet**: Trend and seasonality decomposition with holiday effects  
- **XGBoost**: Machine learning approach with engineered time features  
- Automatic model loading with fallback training on recent data  
- Robust error handling and NaN value management  

### 📈 3. Advanced Visualization  
- Professional dark theme with custom color palette  
- Interactive forecast plots with confidence intervals  
- Model comparison charts with performance metrics  
- Continuous line plotting with gap interpolation  
- Realistic noise addition for natural-looking forecasts  

### 💡 4. Performance Evaluation  
- **Safe metrics calculation** with NaN filtering  
- Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)  
- Model comparison with highlighted best performers  
- Confidence interval visualization for uncertainty quantification  

### 🌐 5. Interactive Dashboard  
- **Streamlit web interface** with:  
  - Adjustable forecast horizon (1-48 hours)  
  - Single model analysis and multi-model comparison  
  - Real-time model performance metrics  
  - Professional styling with enhanced user experience  
  - Responsive design for various screen sizes  

---

## 📊 Key Features  

### 🎯 Forecasting Capabilities  
- **Multi-horizon forecasting**: 1 to 48 hours ahead  
- **Model ensemble**: Three complementary forecasting approaches  
- **Confidence intervals**: Uncertainty quantification for SARIMA  
- **Realistic variations**: Added noise based on historical volatility  

### 🔧 Technical Robustness  
- **NaN handling**: Safe metric calculation with data validation  
- **Model caching**: Efficient loading and reuse of trained models  
- **Error recovery**: Graceful handling of model failures  
- **Data continuity**: Gap filling and interpolation for smooth plots  

### 🎨 User Experience  
- **Dark theme**: Professional appearance with enhanced readability  
- **Interactive controls**: Easy model selection and parameter adjustment  
- **Performance metrics**: Real-time accuracy assessment  
- **Visual clarity**: Clean plots with proper legends and styling  

---

## 📊 Results & Performance  
- **SARIMA**: Excellent for capturing seasonal patterns and trends  
- **Prophet**: Robust handling of holidays and irregular patterns  
- **XGBoost**: Superior performance with engineered time features  
- **Ensemble approach**: Combined insights from multiple methodologies  
- **Real-time evaluation**: Immediate feedback on forecast accuracy  

---

## ✅ Technical Highlights  
This project demonstrates **production-ready time series forecasting**: automated data pipeline, robust model handling, advanced visualization, comprehensive error management, and interactive deployment. The application provides reliable energy consumption predictions with professional-grade user interface and performance monitoring.

---

## 📚 Model Details  

### 🔄 SARIMA (Seasonal AutoRegressive Integrated Moving Average)  
- **Order**: (1,1,1) with seasonal order (1,1,1,24)  
- **Seasonality**: 24-hour daily patterns  
- **Training**: Last 30 days for computational efficiency  
- **Output**: Point forecasts with confidence intervals  

### 📈 Prophet  
- **Components**: Trend, daily/weekly seasonality  
- **Robustness**: Handles missing data and outliers  
- **Flexibility**: Automatic changepoint detection  
- **Output**: Trend decomposition with uncertainty  

### 🌳 XGBoost  
- **Features**: Hour, day-of-week, cyclical encodings, lag variables  
- **Engineering**: Sine/cosine transformations for cyclical patterns  
- **Lags**: 1-hour, 2-hour, and 24-hour historical values  
- **Output**: Non-linear pattern capture with feature importance  

---

## 📊 Evaluation Metrics  
- **MAE (Mean Absolute Error)**: Average prediction error magnitude  
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily  
- **Visual Assessment**: Plot-based evaluation of forecast quality  
- **Comparative Analysis**: Side-by-side model performance  

---

## 🌐 Live App  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://energy-consumption-time-series-forecasting-5skuylt3rhcyjn4muk3.streamlit.app/)


---

## 📚 Useful Links  
- [Household Power Consumption Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)  
- [Statsmodels SARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)  
- [Prophet Documentation](https://facebook.github.io/prophet/)  
- [XGBoost Documentation](https://xgboost.readthedocs.io/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  

---

> 🔖 Submitted as part of the **DevelopersHub Internship Program**
