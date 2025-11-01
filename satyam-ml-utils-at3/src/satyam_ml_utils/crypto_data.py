"""
crypto_data.py
==============
Cryptocurrency data fetching and preprocessing utilities for Ethereum analysis.
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class CryptoDataFetcher:
    """
    Fetches cryptocurrency data from multiple APIs for comprehensive analysis.
    """
    
    def __init__(self):
        """Initialize the crypto data fetcher with API endpoints."""
        self.api_endpoints = {
            'kraken': 'https://api.kraken.com/0/public/OHLC',
            'coingecko': 'https://api.coingecko.com/api/v3/coins/ethereum/ohlc',
            'coindesk': 'https://api.coindesk.com/v1/bpi/historical/close.json'
        }
        
    def fetch_kraken_data(self, symbol: str = 'XETHZUSD', interval: int = 1440, 
                         days: int = 365) -> pd.DataFrame:
        """
        Fetch OHLC data from Kraken API.
        
        Parameters
        ----------
        symbol : str
            Trading pair symbol (default: XETHZUSD for ETH/USD)
        interval : int
            Time interval in minutes (default: 1440 for daily)
        days : int
            Number of days to fetch (default: 365)
        
        Returns
        -------
        pd.DataFrame
            OHLC data with columns: timestamp, open, high, low, close, volume
        """
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp())
            
            params = {
                'pair': symbol,
                'interval': interval,
                'since': since
            }
            
            response = requests.get(self.api_endpoints['kraken'], params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'result' in data and symbol in data['result']:
                ohlc_data = data['result'][symbol]
                
                df = pd.DataFrame(ohlc_data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
                ])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['timeOpen'] = df['timestamp']
                
                # Convert to numeric
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
                
                return df[['timeOpen', 'open', 'high', 'low', 'close', 'volume']]
            else:
                print(f"Warning: No data found for {symbol} in Kraken API")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching Kraken data: {e}")
            return pd.DataFrame()
    
    def fetch_coingecko_data(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch OHLC data from CoinGecko API.
        
        Parameters
        ----------
        days : int
            Number of days to fetch (default: 365)
        
        Returns
        -------
        pd.DataFrame
            OHLC data with columns: timestamp, open, high, low, close
        """
        try:
            params = {'vs_currency': 'usd', 'days': days}
            
            response = requests.get(self.api_endpoints['coingecko'], params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data:
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['timeOpen'] = df['timestamp']
                
                # Add volume (estimated)
                df['volume'] = (df['high'] + df['low']) / 2 * np.random.uniform(0.8, 1.2, len(df))
                
                return df[['timeOpen', 'open', 'high', 'low', 'close', 'volume']]
            else:
                print("Warning: No data found in CoinGecko API")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error fetching CoinGecko data: {e}")
            return pd.DataFrame()
    
    def fetch_combined_data(self, days: int = 365) -> pd.DataFrame:
        """
        Fetch data from multiple sources and combine them.
        
        Parameters
        ----------
        days : int
            Number of days to fetch (default: 365)
        
        Returns
        -------
        pd.DataFrame
            Combined OHLC data from multiple sources
        """
        print("🔄 Fetching cryptocurrency data from multiple sources...")
        
        # Fetch from different sources
        kraken_data = self.fetch_kraken_data(days=days)
        coingecko_data = self.fetch_coingecko_data(days=days)
        
        # Combine data sources
        all_data = []
        
        if not kraken_data.empty:
            kraken_data['source'] = 'kraken'
            all_data.append(kraken_data)
            print(f"✅ Kraken: {len(kraken_data)} records")
        
        if not coingecko_data.empty:
            coingecko_data['source'] = 'coingecko'
            all_data.append(coingecko_data)
            print(f"✅ CoinGecko: {len(coingecko_data)} records")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort by date
            combined_df = combined_df.drop_duplicates(subset=['timeOpen']).sort_values('timeOpen')
            
            # Fill missing values
            combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"✅ Combined dataset: {len(combined_df)} records")
            print(f"📅 Date range: {combined_df['timeOpen'].min()} to {combined_df['timeOpen'].max()}")
            
            return combined_df
        else:
            print("❌ No data available from any source")
            return pd.DataFrame()
    
    def create_sample_data(self, days: int = 365) -> pd.DataFrame:
        """
        Create sample Ethereum data for testing when APIs are unavailable.
        
        Parameters
        ----------
        days : int
            Number of days to generate (default: 365)
        
        Returns
        -------
        pd.DataFrame
            Sample OHLC data
        """
        print("🔄 Creating sample Ethereum data...")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 2000  # Starting ETH price
        
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Generate realistic price movements
            daily_return = np.random.normal(0.001, 0.03)  # 0.1% mean return, 3% volatility
            current_price *= (1 + daily_return)
            
            # Generate OHLC from close price
            volatility = np.random.uniform(0.01, 0.05)
            high = current_price * (1 + np.random.uniform(0, volatility))
            low = current_price * (1 - np.random.uniform(0, volatility))
            open_price = current_price * (1 + np.random.uniform(-volatility/2, volatility/2))
            
            # Ensure OHLC relationships
            high = max(high, open_price, current_price)
            low = min(low, open_price, current_price)
            
            prices.append({
                'timeOpen': dates[i],
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(current_price, 2),
                'volume': np.random.uniform(1000000, 5000000)
            })
        
        df = pd.DataFrame(prices)
        print(f"✅ Sample data created: {len(df)} records")
        print(f"📅 Date range: {df['timeOpen'].min()} to {df['timeOpen'].max()}")
        
        return df


def get_ethereum_data(days: int = 365, use_sample: bool = False) -> pd.DataFrame:
    """
    Get Ethereum data from multiple sources or create sample data.
    
    Parameters
    ----------
    days : int
        Number of days to fetch (default: 365)
    use_sample : bool
        If True, create sample data instead of fetching from APIs
    
    Returns
    -------
    pd.DataFrame
        Ethereum OHLC data
    """
    fetcher = CryptoDataFetcher()
    
    if use_sample:
        return fetcher.create_sample_data(days=days)
    else:
        return fetcher.fetch_combined_data(days=days)


def preprocess_crypto_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess cryptocurrency data for ML modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw cryptocurrency data
    
    Returns
    -------
    pd.DataFrame
        Preprocessed data ready for ML
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ensure datetime column
    if 'timeOpen' in df.columns:
        df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    
    # Sort by date
    df = df.sort_values('timeOpen').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timeOpen']).reset_index(drop=True)
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Add basic features
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['open'] / df['close']
    
    # Add rolling statistics
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        df[f'volatility_{window}'] = df['close'].rolling(window=window).std()
    
    return df
