# 🚀 Ethereum Price Predictor

A comprehensive data product for predicting Ethereum prices using advanced machine learning models. Built as part of AT3 - Data Product with Machine Learning assignment.

## 📋 Project Information

**Student:** Satyam Palkar (25217353)  
**Group:** 25217353  
**Course:** 36120 - Advanced Machine Learning  
**University:** University of Technology Sydney  
**Repository:** [GitHub](https://github.com/SatyamPalkar/eth-streamlit)

## ✨ Features

- 🤖 **Multiple ML Models**: LightGBM, XGBoost, and ElasticNet for price prediction
- 📈 **Interactive Visualizations**: Candlestick charts, volume analysis, and trend indicators
- 🔮 **Real-time Predictions**: Next day HIGH price forecasting with confidence metrics
- 🌐 **FastAPI Integration**: RESTful API support for production deployments
- 📊 **Model Performance Metrics**: MAE, R², RMSE, and direction accuracy
- 🎨 **Modern UI**: Beautiful gradient-based dashboard with Streamlit

## 🏆 Best Model Performance

**LightGBM Model** (Best Performing)
- **MAE**: 0.030501
- **R² Score**: 0.85
- **Direction Accuracy**: 58.5%
- **Model Type**: Gradient Boosting

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/SatyamPalkar/eth-streamlit.git
cd eth-streamlit
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

4. **Access the app** at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Fork this repository on GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your repository
4. Set main file path to `streamlit_app.py`
5. Deploy! 🎉

## 📦 Project Structure

```
eth-streamlit/
├── streamlit_app.py              # Main Streamlit application
├── models/
│   └── ethereum_prediction/      # Trained ML models
│       ├── best_lgbm_model.joblib
│       ├── best_xgb_model.joblib
│       ├── production_pipeline.joblib
│       └── production_metadata.json
├── satyam-ml-utils-at3/          # Custom ML utilities package
├── notebooks/
│   └── ethereum_prediction/      # Jupyter notebooks for analysis
├── reports/                      # Model performance reports
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Dependencies

- **Streamlit** 1.36.0 - Web app framework
- **XGBoost** 2.1.0 - Gradient boosting
- **LightGBM** 4.4.0 - Fast gradient boosting
- **scikit-learn** 1.5.1 - ML utilities
- **pandas** 2.2.2 - Data manipulation
- **plotly** 5.15.0 - Interactive visualizations
- **joblib** 1.4.2 - Model serialization

See `requirements.txt` for complete dependency list.

## 🤖 Model Details

### Features Used (45 total)
- Basic: OHLCV (Open, High, Low, Close, Volume)
- Technical Indicators: RSI, MACD, Bollinger Bands, ATR
- Moving Averages: SMA 20/50, EMA 12/26
- Lag Features: Price and volume lags (1, 3, 7 days)
- Rolling Statistics: Mean, std, correlation windows
- Momentum Indicators: Price momentum, volume-price trend

### Hyperparameters (LightGBM)
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8

## 📊 API Integration

The app supports FastAPI integration for production deployments:

**Example FastAPI endpoint**:
```
POST https://satyam-eth-api.onrender.com/predict
```

**Request payload**:
```json
{
  "close": 2500.0,
  "volume": 15000000,
  "open": 2480.0,
  "high": 2550.0,
  "low": 2470.0,
  "price_change": 0.02,
  "volatility": 0.05,
  "ma_7": 2520.0,
  "ma_30": 2400.0,
  "rsi": 55.0,
  "macd": 0.5,
  "macd_signal": 0.3
}
```

## 📈 Usage Examples

### Making Predictions
1. Launch the app
2. Select date range for historical data
3. Click "🚀 Predict Next Day HIGH Price"
4. View predictions with confidence metrics

### Viewing Performance
1. Check model metrics in sidebar
2. Review performance charts
3. Compare historical vs predicted prices
4. Analyze volatility trends

## 🔬 Model Training

Model training and hyperparameter tuning can be found in:
- `notebooks/ethereum_prediction/36120-25SP-AT3-group_25217353-student_id.ipynb`

## 📝 License

MIT License - Copyright (c) 2025 Satyam Palkar

## 👤 Author

**Satyam Palkar**  
University of Technology Sydney  
Advanced Machine Learning Course  
[GitHub Profile](https://github.com/SatyamPalkar)

## 🙏 Acknowledgments

- CoinGecko API for historical price data
- UTS Advanced ML Course (36120)
- Streamlit team for amazing framework
- LightGBM and XGBoost communities

---

⭐ Star this repo if you find it useful!