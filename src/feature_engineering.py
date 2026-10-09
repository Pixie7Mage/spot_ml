import pandas as pd
import numpy as np

def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes great-circle distance between two GPS coordinates in miles."""
    R = 3958.8  # Earth radius in statute miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def bearing(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Computes initial bearing in degrees from pickup to delivery."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs robust domain and spatial features designed for shift-resistance.
    Excludes non-stationary temporal features (dayofyear, month) and discrete city IDs
    that fail on unseen forward periods and new evaluation nodes.
    """
    df = df.copy()

    # 1. Missingness Indicators
    df["is_missing_weight"] = df["weight"].isna().astype(int) if "weight" in df.columns else 0
    df["is_missing_mkt"] = df["market_index"].isna().astype(int) if "market_index" in df.columns else 0

    # 2. Continuous Spatial & Routing Features (Robust to new cities)
    if {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}.issubset(df.columns):
        df["haversine"] = haversine_distance(
            df["pickup_lat"].values, df["pickup_lon"].values,
            df["delivery_lat"].values, df["delivery_lon"].values
        )
        df["circuity"] = df["distance"] / (df["haversine"] + 1.0)
        df["lat_diff"] = np.abs(df["delivery_lat"] - df["pickup_lat"])
        df["lon_diff"] = np.abs(df["delivery_lon"] - df["pickup_lon"])
        df["bearing"] = bearing(
            df["pickup_lat"].values, df["pickup_lon"].values,
            df["delivery_lat"].values, df["delivery_lon"].values
        )

    # 3. Freight Physics & Fixed-Cost Non-linearities
    if "distance" in df.columns:
        df["log_dist"] = np.log(df["distance"].clip(lower=1.0))
        df["inv_dist"] = 1000.0 / df["distance"].clip(lower=1.0)  # Captures short-haul fixed cost

    if "clean_weight" in df.columns and "distance" in df.columns:
        df["log_weight"] = np.log(df["clean_weight"].clip(lower=100.0))
        df["ton_miles"] = (df["clean_weight"] / 2000.0) * df["distance"]
        df["weight_per_mile"] = df["clean_weight"] / (df["distance"] + 1.0)
        df["is_heavy"] = (df["clean_weight"] > 35000.0).astype(int)

    # 4. Market Signals & Elasticity
    if "clean_mkt" in df.columns:
        df["log_mkt"] = np.log(df["clean_mkt"].clip(lower=0.1))

    if "quote_signal" in df.columns and "distance" in df.columns:
        df["base_quote"] = df["distance"] * df["quote_signal"]
        if "clean_mkt" in df.columns:
            df["mkt_adj_quote"] = df["base_quote"] * df["clean_mkt"]
        if "clean_weight" in df.columns:
            df["rate_per_ton_mile_signal"] = df["base_quote"] / (df["ton_miles"] + 1.0)

    # 5. Invariant Temporal Features (ONLY weekly periodicity - NO dayofyear or month)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["dayofweek"] = df["date"].dt.dayofweek
        df["day"] = df["date"].dt.day
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        
        # 7-day cyclical calendar encoding
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    return df
