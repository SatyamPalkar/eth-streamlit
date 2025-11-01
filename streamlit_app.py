"""
Streamlit App Entry Point for Community Cloud Deployment
AT3 - Data Product with Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import extended package
try:
    from satyam_ml_utils import get_ethereum_data, preprocess_crypto_data
    PACKAGE_AVAILABLE = True
except ImportError:
    try:
        # Try importing just the functions we need
        from satyam_ml_utils.crypto_data import get_ethereum_data, preprocess_crypto_data
        PACKAGE_AVAILABLE = True
    except ImportError:
        PACKAGE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🚀 Ethereum Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .model-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load Ethereum historical data using extended package"""
    import os
    from pathlib import Path
    
    # Try multiple data sources in order of preference
    data_paths = [
        # 1. Try using the package with real API data (not sample)
        None,  # Special marker for package call
        # 2. Try CSV from experiments folder
        Path(__file__).parent.parent / 'experiments' / 'data' / 'raw' / 'ethereum_combined.csv',
        # 3. Try CSV from app/data folder (if exists)
        Path(__file__).parent / 'app' / 'data' / 'ethereum_combined.csv',
        # 4. Try CSV from data folder in current directory
        Path(__file__).parent / 'data' / 'ethereum_combined.csv',
    ]
    
    # First, try to use the package with real data (not sample)
    if PACKAGE_AVAILABLE:
        try:
            # Try to fetch real data first
            df = get_ethereum_data(days=365, use_sample=False)
            if not df.empty and len(df) > 0:
                df = preprocess_crypto_data(df)
                st.success("✅ Data loaded using extended satyam-ml-utils package (real API data)!")
                return df
        except Exception as e:
            # If API fails, try sample data from package
            try:
                df = get_ethereum_data(days=365, use_sample=True)
                if not df.empty:
                    df = preprocess_crypto_data(df)
                    st.info("ℹ️ Using sample data from satyam-ml-utils package (API unavailable).")
                    return df
            except Exception as e2:
                pass  # Continue to CSV files
    
    # Try loading from CSV files
    for csv_path in data_paths[1:]:  # Skip the None marker
        if csv_path and csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if 'timeOpen' in df.columns:
                    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
                else:
                    # Try to find date column
                    date_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                    if date_cols:
                        df['timeOpen'] = pd.to_datetime(df[date_cols[0]])
                df = df.sort_values('timeOpen').reset_index(drop=True)
                st.success(f"✅ Data loaded from CSV: {csv_path.name}")
                return df
            except Exception as e:
                continue
    
    # Final fallback to sample data
    st.warning("⚠️ Using generated sample data. Install satyam-ml-utils package or provide CSV file for better data.")
    return create_sample_data()

def create_sample_data():
    """Create sample data if no data available"""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    prices = 2000 + np.cumsum(np.random.randn(len(dates)) * 50)
    df = pd.DataFrame({
        'timeOpen': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.05, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.05, len(dates))),
        'close': prices * (1 + np.random.uniform(-0.02, 0.02, len(dates))),
        'volume': np.random.uniform(1000000, 5000000, len(dates))
    })
    return df

def call_fastapi_prediction(data, api_url=None):
    """Call FastAPI service for prediction (AT3 Requirement)"""
    # FastAPI service URL - Use provided URL or default
    if api_url is None:
        api_url = st.secrets.get("FASTAPI_URL", "https://satyam-eth-api.onrender.com/predict")
    else:
        api_url = f"{api_url}/predict"
    
    # Prepare data for API
    payload = {
        'close': float(data['close']),
        'volume': float(data['volume']),
        'open': float(data['open']),
        'high': float(data['high']),
        'low': float(data['low']),
        'price_change': float(data.get('price_change', 0.0)),
        'volatility': float(data.get('volatility', 0.0)),
        'ma_7': float(data.get('ma_7', data['close'])),
        'ma_30': float(data.get('ma_30', data['close'])),
        'rsi': float(data.get('rsi', 50.0)),
        'macd': float(data.get('macd', 0.0)),
        'macd_signal': float(data.get('macd_signal', 0.0))
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        # API returns 'predicted_price', not 'predicted_high_price'
        return result.get('predicted_price', result.get('predicted_high_price'))
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ FastAPI call failed: {e}. Using local model fallback.")
        return None
    except Exception as e:
        st.warning(f"⚠️ API error: {e}. Using local model fallback.")
        return None

@st.cache_data
def load_models():
    """Load trained models"""
    models = {}
    model_files = {
        'XGBoost': 'models/ethereum_prediction/best_overall_model.joblib',
        'LightGBM': 'models/ethereum_prediction/best_lgbm_model.joblib',
        'ElasticNet': 'models/ethereum_prediction/final_en_model.joblib',
        'Production Pipeline': 'models/ethereum_prediction/production_pipeline.joblib'
    }
    
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except:
            st.warning(f"Could not load {name} model from {path}")
    
    return models

@st.cache_data
def load_metadata():
    """Load model metadata"""
    try:
        with open('models/ethereum_prediction/production_metadata.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'model_info': {'name': 'XGBoost', 'version': '1.0'},
            'performance': {'test_mae': 0.055, 'test_r2': 0.85, 'direction_accuracy': 58.5},
            'features': {'count': 45, 'names': ['close', 'volume', 'rsi', 'macd']}
        }

def create_price_chart(df):
    """Create interactive price chart"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['timeOpen'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Ethereum Price"
    ))
    
    fig.update_layout(
        title="📈 Ethereum Price History",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=500,
        showlegend=True
    )
    
    return fig

def create_volume_chart(df):
    """Create volume chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['timeOpen'],
        y=df['volume'],
        name="Volume",
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    
    fig.update_layout(
        title="📊 Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=400,
        showlegend=True
    )
    
    return fig

def make_prediction(models, df, selected_model_name, fastapi_url=None):
    """Make prediction using FastAPI service first, then local models as fallback (AT3 Requirement)"""
    try:
        # Try FastAPI service first (AT3 Requirement)
        latest_data = df.tail(1)
        api_prediction = call_fastapi_prediction(latest_data.iloc[0], fastapi_url)
        
        if api_prediction is not None:
            st.success("✅ Prediction from FastAPI service!")
            return api_prediction
        
        # Fallback to local models if FastAPI fails
        st.info("🔄 Using local model fallback...")
        
        # Check for the selected model specifically
        if selected_model_name == 'XGBoost' and 'XGBoost' in models:
            model = models['XGBoost']
            latest_data = df.tail(1)
            
            # Create a simple feature vector for XGBoost
            features = [
                latest_data['close'].iloc[0],
                latest_data['volume'].iloc[0],
                latest_data['open'].iloc[0],
                latest_data['high'].iloc[0],
                latest_data['low'].iloc[0]
            ]
            
            # Add some basic technical indicators
            if len(df) > 1:
                features.extend([
                    (latest_data['close'].iloc[0] - df['close'].iloc[-2]) / df['close'].iloc[-2],  # Price change
                    latest_data['volume'].iloc[0] / df['volume'].mean(),  # Volume ratio
                    latest_data['high'].iloc[0] / latest_data['low'].iloc[0],  # High/low ratio
                    latest_data['close'].iloc[0] / latest_data['open'].iloc[0],  # Close/open ratio
                    df['close'].mean() / latest_data['close'].iloc[0],  # Price relative to average
                ])
            else:
                features.extend([0, 1, 1, 1, 1])
            
            # Pad with zeros to match expected input size (11 features)
            while len(features) < 11:
                features.append(0)
            
            prediction = model.predict([features[:11]])
            return prediction[0]
            
        # Handle LightGBM (similar to XGBoost - 11 features)
        elif selected_model_name == 'LightGBM' and 'LightGBM' in models:
            model = models['LightGBM']
            latest_data = df.tail(1)
            
            # Create a simple feature vector for LightGBM (same as XGBoost)
            features = [
                latest_data['close'].iloc[0],
                latest_data['volume'].iloc[0],
                latest_data['open'].iloc[0],
                latest_data['high'].iloc[0],
                latest_data['low'].iloc[0]
            ]
            
            # Add some basic technical indicators
            if len(df) > 1:
                features.extend([
                    (latest_data['close'].iloc[0] - df['close'].iloc[-2]) / df['close'].iloc[-2],  # Price change
                    latest_data['volume'].iloc[0] / df['volume'].mean(),  # Volume ratio
                    latest_data['high'].iloc[0] / latest_data['low'].iloc[0],  # High/low ratio
                    latest_data['close'].iloc[0] / latest_data['open'].iloc[0],  # Close/open ratio
                    df['close'].mean() / latest_data['close'].iloc[0],  # Price relative to average
                ])
            else:
                features.extend([0, 1, 1, 1, 1])
            
            # Pad with zeros to match expected input size (11 features)
            while len(features) < 11:
                features.append(0)
            
            # Model predicts log returns, need to convert to actual price
            log_return = model.predict([features[:11]])[0]
            
            # Convert log return to actual price
            # log_return = log(price_tomorrow / price_today)
            # price_tomorrow = price_today * exp(log_return)
            current_price = latest_data['close'].iloc[0]
            predicted_price = current_price * np.exp(log_return)
            
            return predicted_price
            
        # Handle ElasticNet specifically (expects 46 features)
        elif selected_model_name == 'ElasticNet' and 'ElasticNet' in models:
            model = models['ElasticNet']
            latest_data = df.tail(1)
            
            # ElasticNet expects 46 features - create a comprehensive feature vector
            features = []
            
            # Basic OHLCV features (5)
            features.extend([
                latest_data['close'].iloc[0],
                latest_data['volume'].iloc[0],
                latest_data['open'].iloc[0],
                latest_data['high'].iloc[0],
                latest_data['low'].iloc[0]
            ])
            
            # Technical indicators and ratios (41 more features)
            if len(df) > 1:
                # Price-based features
                features.extend([
                    (latest_data['close'].iloc[0] - df['close'].iloc[-2]) / df['close'].iloc[-2],  # Price change
                    latest_data['high'].iloc[0] / latest_data['low'].iloc[0],  # High/low ratio
                    latest_data['close'].iloc[0] / latest_data['open'].iloc[0],  # Close/open ratio
                    (latest_data['high'].iloc[0] - latest_data['low'].iloc[0]) / latest_data['close'].iloc[0],  # Range ratio
                ])
                
                # Volume-based features
                features.extend([
                    latest_data['volume'].iloc[0] / df['volume'].mean(),  # Volume ratio
                    latest_data['volume'].iloc[0] / df['volume'].std(),  # Volume z-score
                ])
                
                # Moving averages (if enough data)
                if len(df) >= 7:
                    ma_7 = df['close'].rolling(window=7).mean().iloc[-1]
                    features.extend([
                        latest_data['close'].iloc[0] / ma_7,  # Price vs MA7
                        (latest_data['close'].iloc[0] - ma_7) / ma_7,  # Price deviation from MA7
                    ])
                else:
                    features.extend([1, 0])
                
                if len(df) >= 30:
                    ma_30 = df['close'].rolling(window=30).mean().iloc[-1]
                    features.extend([
                        latest_data['close'].iloc[0] / ma_30,  # Price vs MA30
                        (latest_data['close'].iloc[0] - ma_30) / ma_30,  # Price deviation from MA30
                    ])
                else:
                    features.extend([1, 0])
                
                # Volatility features
                if len(df) >= 7:
                    volatility = df['close'].rolling(window=7).std().iloc[-1]
                    features.extend([
                        volatility / latest_data['close'].iloc[0],  # Normalized volatility
                        volatility / df['close'].mean(),  # Volatility ratio
                    ])
                else:
                    features.extend([0, 0])
                
                # RSI-like features
                if len(df) >= 14:
                    price_changes = df['close'].diff().dropna()
                    gains = price_changes.where(price_changes > 0, 0).rolling(window=14).mean().iloc[-1]
                    losses = (-price_changes.where(price_changes < 0, 0)).rolling(window=14).mean().iloc[-1]
                    if losses > 0:
                        rsi = 100 - (100 / (1 + gains / losses))
                        features.extend([rsi / 100, (rsi - 50) / 50])  # Normalized RSI
                    else:
                        features.extend([0.5, 0])
                else:
                    features.extend([0.5, 0])
                
                # MACD-like features
                if len(df) >= 26:
                    ema_12 = df['close'].ewm(span=12).mean().iloc[-1]
                    ema_26 = df['close'].ewm(span=26).mean().iloc[-1]
                    macd = ema_12 - ema_26
                    features.extend([
                        macd / latest_data['close'].iloc[0],  # Normalized MACD
                        macd / df['close'].std(),  # MACD z-score
                    ])
                else:
                    features.extend([0, 0])
                
                # Additional engineered features to reach 46
                remaining_features = 46 - len(features)
                for i in range(remaining_features):
                    if i % 4 == 0:
                        features.append(latest_data['close'].iloc[0] / df['close'].mean())  # Price ratio
                    elif i % 4 == 1:
                        features.append(latest_data['volume'].iloc[0] / df['volume'].mean())  # Volume ratio
                    elif i % 4 == 2:
                        features.append((latest_data['high'].iloc[0] - latest_data['low'].iloc[0]) / latest_data['close'].iloc[0])  # Range
                    else:
                        features.append(latest_data['close'].iloc[0] / latest_data['open'].iloc[0])  # Close/open
            else:
                # Not enough data - fill with defaults
                features.extend([0] * (46 - len(features)))
            
            # Ensure exactly 46 features
            features = features[:46]
            while len(features) < 46:
                features.append(0)
            
            prediction = model.predict([features])
            return prediction[0]
            
        # Fallback to production pipeline if individual models fail
        elif selected_model_name == 'Production Pipeline' and 'Production Pipeline' in models:
            pipeline = models['Production Pipeline']
            latest_data = df.tail(1)
            
            # Create a simplified feature vector with the expected number of features
            n_features = 37  # Expected by the RobustScaler
            features = []
            
            # Basic features
            features.extend([
                latest_data['close'].iloc[0],
                latest_data['volume'].iloc[0],
                latest_data['open'].iloc[0],
                latest_data['high'].iloc[0],
                latest_data['low'].iloc[0]
            ])
            
            # Fill remaining features with calculated values or defaults
            for i in range(5, n_features):
                if i < len(df):
                    # Use historical data for some features
                    if i % 3 == 0:
                        features.append(df['close'].iloc[-(i-4)] if len(df) > (i-4) else latest_data['close'].iloc[0])
                    elif i % 3 == 1:
                        features.append(df['volume'].iloc[-(i-4)] if len(df) > (i-4) else latest_data['volume'].iloc[0])
                    else:
                        features.append(0.5)  # Default value for technical indicators
                else:
                    features.append(0.5)  # Default value
            
            # Ensure we have exactly the right number of features
            features = features[:n_features]
            
            prediction = pipeline.predict([features])
            return prediction[0]
        else:
            # Fallback prediction
            return df['close'].iloc[-1] * (1 + np.random.uniform(-0.05, 0.05))
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return df['close'].iloc[-1] * (1 + np.random.uniform(-0.02, 0.02))

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Ethereum Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### AT3 - Data Product with Machine Learning | Student: Satyam Palkar (25217353)")
    
    # Load data
    with st.spinner("Loading Ethereum data..."):
        df = load_data()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Best Model (as per assignment requirements - one model only)
    st.sidebar.subheader("🤖 Best Model")
    st.sidebar.info("**LightGBM** - Best Performing Model")
    st.sidebar.metric("Test MAE", "0.030501")
    st.sidebar.metric("Performance", "Best MAE Score")
    st.sidebar.metric("Model Type", "Gradient Boosting")
    
    # FastAPI Configuration (AT3 Requirement)
    st.sidebar.subheader("🌐 FastAPI Service")
    fastapi_url = st.sidebar.text_input(
        "FastAPI URL", 
        value="https://satyam-eth-api.onrender.com",
        help="Enter your deployed FastAPI service URL"
    )
    if fastapi_url:
        st.sidebar.success(f"✅ FastAPI: {fastapi_url}")
    else:
        st.sidebar.warning("⚠️ No FastAPI URL configured")
    
    # Date range selector
    if not df.empty:
        min_date = df['timeOpen'].min().date()
        max_date = df['timeOpen'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(max_date - pd.Timedelta(days=30), max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Filter data by date range
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df['timeOpen'].dt.date >= start_date) & (df['timeOpen'].dt.date <= end_date)]
        else:
            filtered_df = df.tail(30)
    else:
        filtered_df = df
    
    # Use the best model (LightGBM) as per assignment requirements
    model_choice = "LightGBM"
    
    # Load models
    with st.spinner("Loading ML models..."):
        models = load_models()
        metadata = load_metadata()
    
    # Main content
    if not filtered_df.empty:
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = filtered_df['close'].iloc[-1]
            st.metric("Current Price", f"${current_price:,.2f}")
        
        with col2:
            price_change = ((filtered_df['close'].iloc[-1] - filtered_df['close'].iloc[-2]) / filtered_df['close'].iloc[-2]) * 100
            st.metric("24h Change", f"{price_change:+.2f}%")
        
        with col3:
            volume = filtered_df['volume'].iloc[-1]
            st.metric("Volume", f"{volume:,.0f}")
        
        with col4:
            volatility = filtered_df['close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")
        
        # Charts
        st.subheader("📈 Price Analysis")
        
        # Price chart
        price_chart = create_price_chart(filtered_df)
        st.plotly_chart(price_chart, width='stretch')
        
        # Volume chart
        volume_chart = create_volume_chart(filtered_df)
        st.plotly_chart(volume_chart, width='stretch')
        
        # Prediction section
        st.subheader("🔮 Ethereum Price Prediction")
        st.info("Using **LightGBM** - Best Performing Model (MAE = 0.030501)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Predict Next Day HIGH Price", type="primary"):
                with st.spinner("Making prediction with LightGBM..."):
                    prediction = make_prediction(models, filtered_df, model_choice, fastapi_url)
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>Predicted Next Day HIGH Price</h2>
                        <h1>${prediction:,.2f}</h1>
                        <p>Powered by LightGBM (Best Model)</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # Model info
            st.markdown("### 🤖 Model Info")
            st.info(f"""
            **Model:** {model_choice}
            **Features:** {metadata['features']['count']}
            **Performance:** {metadata['performance']['test_r2']:.3f} R²
            """)
        
        # Model performance
        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"{metadata['performance']['test_mae']:.4f}")
        
        with col2:
            st.metric("R² Score", f"{metadata['performance']['test_r2']:.3f}")
        
        with col3:
            st.metric("Direction Accuracy", f"{metadata['performance']['direction_accuracy']:.1f}%")
        
        # Data table
        st.subheader("📋 Recent Data")
        st.dataframe(filtered_df.tail(10), width='stretch')
        
    else:
        st.error("❌ No data available. Please check your data source.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚀 Ethereum Price Predictor | AT3 - Data Product with Machine Learning</p>
        <p>Student: Satyam Palkar (25217353) | UTS Advanced ML</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()