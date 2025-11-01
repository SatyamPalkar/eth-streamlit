#!/usr/bin/env python3
"""
Test script for Streamlit ↔ FastAPI connection
AT3 - Data Product with Machine Learning
"""

import requests
import json
import sys

def test_fastapi_connection(api_url="http://localhost:8000"):
    """Test FastAPI service connection and endpoints"""
    
    print("🧪 Testing FastAPI Connection...")
    print("=" * 40)
    
    # Test 1: Health check
    try:
        print("1. Testing health endpoint...")
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Health check passed")
            print(f"   📊 Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
    
    # Test 2: Model info
    try:
        print("\n2. Testing model info endpoint...")
        response = requests.get(f"{api_url}/model/info", timeout=5)
        if response.status_code == 200:
            print("   ✅ Model info retrieved")
            print(f"   📊 Model: {response.json().get('name', 'Unknown')}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Model info error: {e}")
    
    # Test 3: Prediction
    try:
        print("\n3. Testing prediction endpoint...")
        payload = {
            "close": 3000.0,
            "volume": 1000000.0,
            "open": 2990.0,
            "high": 3010.0,
            "low": 2980.0,
            "price_change": 0.01,
            "volatility": 0.02,
            "ma_7": 3000.0,
            "ma_30": 3000.0,
            "rsi": 50.0,
            "macd": 0.0,
            "macd_signal": 0.0
        }
        
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            predicted_price = result.get('predicted_high_price', 0)
            print("   ✅ Prediction successful")
            print(f"   📊 Predicted price: ${predicted_price:,.2f}")
            
            # Check if prediction is realistic
            if 1000 < predicted_price < 50000:
                print("   ✅ Prediction looks realistic")
            else:
                print("   ⚠️ Prediction might be unrealistic")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   📊 Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
    
    print("\n" + "=" * 40)
    print("🎯 Test completed!")

if __name__ == "__main__":
    # Get API URL from command line or use default
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_fastapi_connection(api_url)


