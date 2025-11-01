"""
Test script to check FastAPI service endpoints
"""
import requests
import json

API_URL = "https://satyam-eth-api.onrender.com"

def test_health():
    """Test the health endpoint"""
    print("=" * 50)
    print("Testing Health Endpoint")
    print("=" * 50)
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Status: {data.get('status')}")
        print(f"✅ Timestamp: {data.get('timestamp')}")
        print(f"✅ Model Loaded: {data.get('model_loaded')}")
        print(f"✅ Data Available: {data.get('data_available')}")
        print(f"\nFull Response:\n{json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    print("\n" + "=" * 50)
    print("Testing Prediction Endpoint")
    print("=" * 50)
    
    # Sample Ethereum data
    payload = {
        "close": 2500.0,
        "volume": 1000000,
        "open": 2480.0,
        "high": 2520.0,
        "low": 2470.0,
        "price_change": 0.02,
        "volatility": 0.05,
        "ma_7": 2480.0,
        "ma_30": 2450.0,
        "rsi": 55.0,
        "macd": 10.0,
        "macd_signal": 8.0
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        print(f"✅ Prediction Successful!")
        print(f"✅ Predicted Price: ${data.get('predicted_price', 'N/A'):,.2f}")
        print(f"✅ Confidence Score: {data.get('confidence_score', 'N/A')}")
        print(f"✅ Model Used: {data.get('model_used', 'N/A')}")
        print(f"✅ Features Count: {data.get('features_count', 'N/A')}")
        print(f"✅ Prediction Date: {data.get('prediction_date', 'N/A')}")
        print(f"\nFull Response:\n{json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text}")
        return False

if __name__ == "__main__":
    print(f"\n🚀 Testing FastAPI Service: {API_URL}\n")
    
    health_ok = test_health()
    prediction_ok = test_prediction()
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Health Endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Prediction Endpoint: {'✅ PASS' if prediction_ok else '❌ FAIL'}")
    print("=" * 50)

