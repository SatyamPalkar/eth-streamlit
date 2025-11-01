"""
preprocessing.py
================
Data preprocessing utilities for feature engineering and target creation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def handle_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Fill missing values in numeric columns using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strategy : str, default="mean"
        Strategy for imputation ("mean", "median", "most_frequent", "constant").

    Returns
    -------
    df : pd.DataFrame
        DataFrame with missing values filled.
    """
    imputer = SimpleImputer(strategy=strategy)
    df[df.columns] = imputer.fit_transform(df)
    return df


def scale_features(df: pd.DataFrame, numeric_cols=None) -> pd.DataFrame:
    """
    Scale numeric features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    numeric_cols : list, optional
        List of numeric columns to scale. If None, all numeric columns are scaled.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with scaled features.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Expand a date column into year, month, day, weekday features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with new columns: year, month, day, weekday.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday
    return df


def make_target_rain(df: pd.DataFrame, date_col: str, precip_col: str, horizon: int = 7) -> pd.Series:
    """
    Create target variable for classification: will it rain in +horizon days?

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of date column (must be sorted).
    precip_col : str
        Name of precipitation column.
    horizon : int, default=7
        Days ahead to check.

    Returns
    -------
    pd.Series
        Binary target (1 if rain in +horizon days, else 0).
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    df["future_precip"] = df[precip_col].shift(-horizon)
    return (df["future_precip"] > 0).astype(int)


def make_target_precip(df: pd.DataFrame, date_col: str, precip_col: str, horizon: int = 3) -> pd.Series:
    """
    Create target variable for regression: cumulative precipitation in next horizon days.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of date column (must be sorted).
    precip_col : str
        Name of precipitation column.
    horizon : int, default=3
        Days ahead for cumulative precipitation.

    Returns
    -------
    pd.Series
        Numeric target for cumulative precipitation.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    return (
        df[precip_col]
        .shift(-1)
        .rolling(window=horizon, min_periods=1)
        .sum()
        .shift(-(horizon - 1))
    )

import pandas as pd
import numpy as np

def preprocess_rain(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing for Rain-or-Not pipeline."""
    X = X.copy()
    if "time" in X.columns:
        X["time"] = pd.to_datetime(X["time"])
        X["month"] = X["time"].dt.month
        X["weekday"] = X["time"].dt.weekday
    return X[["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "month", "weekday"]]


def preprocess_precip(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "time" in X.columns:
        X["time"] = pd.to_datetime(X["time"])
        X["month"] = X["time"].dt.month
        X["weekday"] = X["time"].dt.weekday

    X["temp_range"] = X["temperature_2m_max"] - X["temperature_2m_min"]
    X["temp_max_3d"] = X["temperature_2m_max"].rolling(3, min_periods=1).mean()
    X["temp_min_3d"] = X["temperature_2m_min"].rolling(3, min_periods=1).mean()
    X["temp_range_7d"] = X["temp_range"].rolling(7, min_periods=1).mean()
    X["rain_yesterday"] = (X["precipitation_sum"].shift(1) > 0).astype(int)
    X["precip_3d"] = X["precipitation_sum"].rolling(3, min_periods=1).sum()
    X["precip_7d"] = X["precipitation_sum"].rolling(7, min_periods=1).sum()
    X["precip_avg_3d"] = X["precipitation_sum"].rolling(3, min_periods=1).mean()
    X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
    X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 7)
    X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 7)

    # Drop datetime to avoid DTypePromotionError
    return X.drop(columns=["time"]).dropna().reset_index(drop=True)