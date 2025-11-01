# Satyam ML Utils v0.3.0

A comprehensive Python package for machine learning utilities, now extended with cryptocurrency analysis capabilities for Ethereum price prediction.

## 🚀 **New in v0.3.0 - Cryptocurrency Analysis**

### **Features Added:**
- **Ethereum Price Prediction**: Advanced ML models for predicting Ethereum prices
- **Multi-API Data Integration**: Kraken, CoinGecko, TokenMetrics, CoinDesk
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **FastAPI Integration**: RESTful API for model serving
- **Advanced Feature Engineering**: Lag features, rolling statistics, momentum indicators

### **Crypto Modules:**
- `crypto_data.py`: Data fetching and preprocessing
- `crypto_models.py`: ML models for price prediction
- `crypto_api.py`: FastAPI endpoints for model serving

## 📦 **Installation**

```bash
# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ satyam-ml-utils

# Or install from source
git clone https://github.com/SatyamPalkar/satyam-ml-utils.git
cd satyam-ml-utils
pip install -e .
```

## 🔧 **Dependencies**

### **Core ML Libraries:**
- pandas >= 2.2.2
- numpy >= 1.26.0
- scikit-learn >= 1.5.1
- xgboost >= 2.1.0
- lightgbm >= 4.4.0

### **New Crypto Dependencies:**
- requests >= 2.31.0
- fastapi >= 0.111.0
- uvicorn >= 0.30.1
- streamlit >= 1.36.0
- plotly >= 5.15.0

## 🎯 **Usage Examples**

### **1. Ethereum Data Fetching**
```python
from satyam_ml_utils import get_ethereum_data, preprocess_crypto_data

# Fetch Ethereum data
eth_data = get_ethereum_data(days=365, use_sample=True)

# Preprocess for ML
processed_data = preprocess_crypto_data(eth_data)
print(f"Data shape: {processed_data.shape}")
```

### **2. Ethereum Price Prediction**
```python
from satyam_ml_utils import train_ethereum_model, predict_ethereum_price

# Train model
model = train_ethereum_model(processed_data, save_path="models/eth_model.joblib")

# Make prediction
predicted_price = predict_ethereum_price("models/eth_model.joblib", latest_data)
print(f"Predicted ETH price: ${predicted_price:.2f}")
```

### **3. Advanced Model Training**
```python
from satyam_ml_utils import EthereumPredictor

# Initialize predictor
predictor = EthereumPredictor()

# Create features
df_with_features = predictor.create_features(eth_data)

# Prepare training data
X, y = predictor.prepare_training_data(df_with_features)

# Train multiple models
results = predictor.train_models(X, y)
print(f"Best model: {predictor.best_model_name}")
```

### **4. FastAPI Integration**
```python
from satyam_ml_utils import crypto_api_app
import uvicorn

# Run the API server
uvicorn.run(crypto_api_app, host="0.0.0.0", port=8000)
```

## 🌐 **API Endpoints**

### **Health & Info:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /model/info` - Model information

### **Predictions:**
- `POST /predict` - Predict Ethereum price
- `GET /data/latest` - Latest Ethereum data
- `GET /data/history` - Historical data

### **Model Management:**
- `POST /model/retrain` - Retrain model
- `GET /features/importance` - Feature importance

## 📊 **Model Performance**

### **Supported Algorithms:**
- **XGBoost**: Gradient boosting for complex patterns
- **LightGBM**: Fast gradient boosting
- **ElasticNet**: Linear regression with regularization

### **Evaluation Metrics:**
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **Direction Accuracy**: Up/Down prediction accuracy

## 🔧 **Development**

### **Package Structure:**
```
satyam_ml_utils/
├── src/
│   └── satyam_ml_utils/
│       ├── __init__.py
│       ├── crypto_data.py      # Data fetching & preprocessing
│       ├── crypto_models.py    # ML models & training
│       ├── crypto_api.py       # FastAPI endpoints
│       ├── models.py           # Weather prediction (AT2)
│       ├── features.py        # Feature engineering
│       ├── metrics.py         # Evaluation metrics
│       └── ...
├── tests/
├── pyproject.toml
└── README.md
```

### **Building & Publishing:**
```bash
# Build package
poetry build

# Publish to TestPyPI
poetry publish --repository testpypi
```

## 🎯 **AT3 Integration**

This package is designed for **AT3 - Data Product with Machine Learning** and includes:

1. **Experimentation Phase**: Use in Jupyter notebooks for model development
2. **Streamlit App**: Integrate with Streamlit for data visualization
3. **FastAPI Deployment**: Deploy models as RESTful APIs
4. **Production Pipeline**: Complete ML pipeline for deployment

## 📈 **Performance Features**

- **Time Series Aware**: Chronological train-test splits
- **Feature Engineering**: 20+ technical indicators
- **Robust Scaling**: Handles outliers in crypto data
- **Multiple Models**: Ensemble approach for better predictions
- **API Ready**: FastAPI integration for production deployment

## 🔄 **Backward Compatibility**

All existing AT2 weather prediction functions remain available:
- `predict_rain_smart()`
- `predict_precipitation_smart()`
- `get_api_status()`
- And more...

## 📝 **License**

This package is developed for educational purposes as part of UTS Advanced ML coursework.

## 👨‍💻 **Author**

**Satyam Palkar**  
Student ID: 25217353  
Email: satyam.g.palkar@student.uts.edu.au  
Course: 36120 - Advanced Machine Learning  
University of Technology Sydney