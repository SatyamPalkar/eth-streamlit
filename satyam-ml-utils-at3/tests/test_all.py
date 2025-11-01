import pytest
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression, LinearRegression

from satyam_ml_utils.utils import seed_everything, Timer
from satyam_ml_utils.io import save_model, load_model, save_metrics
from satyam_ml_utils.metrics import evaluate_classification, evaluate_regression
from satyam_ml_utils.preprocessing import handle_missing, scale_features, add_date_features, make_target_rain, make_target_precip
from satyam_ml_utils.models import train_classification, train_regression, predict_rain, predict_precipitation
from satyam_ml_utils.preprocessing import preprocess_rain, preprocess_precip



def test_seed_everything():
    seed_everything(123)
    a = np.random.rand(3)
    seed_everything(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)


def test_timer_context():
    with Timer("dummy"):
        x = sum(range(1000))
    assert x > 0


def test_save_and_load_model(tmp_path):
    model = LogisticRegression()
    path = tmp_path / "model.pkl"
    save_model(model, path)
    loaded = load_model(path)
    assert isinstance(loaded, LogisticRegression)


def test_save_metrics(tmp_path):
    metrics = {"accuracy": 0.9}
    path = tmp_path / "metrics.json"
    save_metrics(metrics, path)
    assert os.path.exists(path)


def test_evaluate_classification():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X, y)
    results = evaluate_classification(model, X, y)
    assert "accuracy" in results and 0 <= results["accuracy"] <= 1


def test_evaluate_regression():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegression().fit(X, y)
    results = evaluate_regression(model, X, y)
    assert "rmse" in results and results["rmse"] >= 0


def test_handle_missing():
    df = pd.DataFrame({"a": [1, np.nan, 3]})
    df_clean = handle_missing(df)
    assert not df_clean.isnull().any().any()


def test_scale_features():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df_scaled = scale_features(df)
    assert np.allclose(df_scaled.mean().round(), 0)


def test_add_date_features():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"]})
    df = add_date_features(df, "date")
    assert "weekday" in df.columns


def test_make_target_rain():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5), "precip": [0, 0, 5, 0, 10]})
    y = make_target_rain(df, "date", "precip", horizon=2)
    assert set(y.unique()).issubset({0, 1})


def test_make_target_precip():
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5), "precip": [0, 1, 2, 3, 4]})
    y = make_target_precip(df, "date", "precip", horizon=2)
    assert isinstance(y, pd.Series)


def test_train_classification_and_predict():
    X = pd.DataFrame({
        "temp": [20, 22, 30, 15, 18, 25],
        "humidity": [60, 65, 70, 55, 58, 62]
    })
    y = [0, 0, 1, 1, 0, 1]
    model, results = train_classification(X, y, model_type="logistic")
    assert "accuracy" in results
    pred = predict_rain(model, [25, 60])
    assert isinstance(pred, bool)



def test_train_regression_and_predict():
    X = pd.DataFrame({"temp": [20, 22, 30, 15], "humidity": [60, 65, 70, 55]})
    y = [5.0, 6.0, 15.0, 7.0]
    model, results = train_regression(X, y, model_type="linear")
    assert "rmse" in results
    pred = predict_precipitation(model, [25, 60])
    assert isinstance(pred, float)

def test_preprocess_rain():
    df = pd.DataFrame([{
        "time": "2023-01-01",
        "temperature_2m_max": 30,
        "temperature_2m_min": 20,
        "precipitation_sum": 5
    }])
    result = preprocess_rain(df)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "month", "weekday"
    }
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "weekday"] == 6


def test_preprocess_precip():
    import pandas as pd
    import numpy as np
    from satyam_ml_utils.preprocessing import preprocess_precip

    df = pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=7),
        "temperature_2m_max": np.linspace(25, 30, 7),
        "temperature_2m_min": np.linspace(15, 20, 7),
        "precipitation_sum": np.arange(7)
    })

    result = preprocess_precip(df)

    expected_cols = {
        "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
        "month", "weekday", "temp_range", "temp_max_3d", "temp_min_3d",
        "temp_range_7d", "rain_yesterday", "precip_3d", "precip_7d",
        "precip_avg_3d", "month_sin", "month_cos", "weekday_sin", "weekday_cos"
    }

    assert isinstance(result, pd.DataFrame)
    assert expected_cols.issubset(set(result.columns))
    assert not result.isnull().values.any()

