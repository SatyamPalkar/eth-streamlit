"""
crypto_models.py
================
Cryptocurrency ML models for Ethereum price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class EthereumPredictor:
    """
    Ethereum price prediction model with multiple algorithms.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the Ethereum predictor."""
        self.random_state = random_state
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for Ethereum price prediction.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw cryptocurrency data
        
        Returns
        -------
        pd.DataFrame
            Data with engineered features
        """
        df = df.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        
        # Lag features
        for lag in [1, 3, 7, 14]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        rolling_mean_20 = df['close'].rolling(window=20).mean()
        rolling_std_20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean_20 + (2 * rolling_std_20)
        df['bb_lower'] = rolling_mean_20 - (2 * rolling_std_20)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Volume indicators
        df['volume_price_trend'] = df['volume'] * df['price_change']
        df['volume_sma_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'high') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training the model.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data with features
        target_col : str
            Target column name (default: 'high')
        
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Features and target for training
        """
        # Create target variable (next day's high price)
        df['target'] = df[target_col].shift(-1)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target'])
        
        # Select feature columns (exclude non-numeric and target columns)
        exclude_cols = ['timeOpen', 'timestamp', 'target', 'source']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        # Remove columns with too many NaN values
        feature_cols = [col for col in feature_cols if df[col].isnull().sum() < len(df) * 0.5]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].fillna(0)
        y = df['target']
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train multiple models for Ethereum price prediction.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        test_size : float
            Proportion of data for testing
        
        Returns
        -------
        Dict[str, Any]
            Training results and model performance
        """
        print("🔄 Training Ethereum prediction models...")
        
        # Time series split for chronological data
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split data chronologically
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        # Initialize models
        models = {
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'ElasticNet': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=self.random_state
            )
        }
        
        results = {}
        
        # Train each model
        for name, model in models.items():
            print(f"  📊 Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Direction accuracy
            train_direction = np.mean(np.sign(y_train.values) == np.sign(y_pred_train)) * 100
            test_direction = np.mean(np.sign(y_test.values) == np.sign(y_pred_test)) * 100
            
            results[name] = {
                'model': model,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_direction': train_direction,
                'test_direction': test_direction,
                'predictions': {
                    'train': y_pred_train,
                    'test': y_pred_test
                }
            }
            
            print(f"    ✅ {name} - Test MAE: {test_mae:.4f}, R²: {test_r2:.3f}, Direction: {test_direction:.1f}%")
        
        # Select best model
        best_model_name = min(results.keys(), key=lambda x: results[x]['test_mae'])
        best_model = results[best_model_name]['model']
        
        print(f"🏆 Best model: {best_model_name}")
        
        # Store models and results
        self.models = {name: results[name]['model'] for name in results.keys()}
        self.best_model = best_model
        self.best_model_name = best_model_name
        self.is_trained = True
        
        return {
            'results': results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best trained model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction
        
        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure we have the right features
        X_features = X[self.feature_columns].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Make prediction
        prediction = self.best_model.predict(X_scaled)
        
        return prediction
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'models': self.models
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.models = model_data['models']
        self.is_trained = True
        
        print(f"✅ Model loaded from {filepath}")


def train_ethereum_model(df: pd.DataFrame, save_path: Optional[str] = None) -> EthereumPredictor:
    """
    Train a complete Ethereum prediction model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Ethereum data with features
    save_path : str, optional
        Path to save the trained model
    
    Returns
    -------
    EthereumPredictor
        Trained model
    """
    predictor = EthereumPredictor()
    
    # Create features
    df_with_features = predictor.create_features(df)
    
    # Prepare training data
    X, y = predictor.prepare_training_data(df_with_features)
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Save model if path provided
    if save_path:
        predictor.save_model(save_path)
    
    return predictor


def predict_ethereum_price(model_path: str, df: pd.DataFrame) -> float:
    """
    Predict Ethereum price using a trained model.
    
    Parameters
    ----------
    model_path : str
        Path to the trained model
    df : pd.DataFrame
        Latest data for prediction
    
    Returns
    -------
    float
        Predicted price
    """
    predictor = EthereumPredictor()
    predictor.load_model(model_path)
    
    # Create features for latest data
    df_with_features = predictor.create_features(df)
    
    # Get latest row for prediction
    latest_data = df_with_features.tail(1)
    
    # Make prediction
    prediction = predictor.predict(latest_data)
    
    return float(prediction[0])
