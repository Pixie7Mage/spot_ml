import pandas as pd
import numpy as np
from typing import Optional, Dict

def clean_weights(df: pd.DataFrame, equipment_medians: Optional[Dict[str, float]] = None) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Cleans cargo weights:
    1. Fixes erroneous negative weights using absolute value.
    2. Imputes missing weights using equipment-specific medians.
    """
    df = df.copy()
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df["clean_weight"] = df["weight"].abs()
        
        if equipment_medians is None:
            equipment_medians = df.groupby("equipment")["clean_weight"].median().to_dict()
            global_median = df["clean_weight"].median()
            if pd.isna(global_median):
                global_median = 30000.0
            equipment_medians["_global_"] = global_median

        fallback = equipment_medians.get("_global_", 30000.0)
        df["clean_weight"] = df.apply(
            lambda r: equipment_medians.get(r["equipment"], fallback) if pd.isna(r["clean_weight"]) else r["clean_weight"],
            axis=1
        )
    return df, equipment_medians


def clean_market_index(df: pd.DataFrame, daily_means: Optional[Dict[str, float]] = None) -> tuple[pd.DataFrame, Dict[str, float]]:
    """
    Cleans market index:
    1. Missing market_index values are imputed using the date's mean market index.
    2. Fallback to global median if the entire day is missing.
    """
    df = df.copy()
    if "market_index" in df.columns:
        df["market_index"] = pd.to_numeric(df["market_index"], errors="coerce")
        
        if daily_means is None:
            # Group by date string
            date_str = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            daily_means = df.groupby(date_str)["market_index"].mean().to_dict()
            global_mkt = df["market_index"].median()
            daily_means["_global_"] = global_mkt if not pd.isna(global_mkt) else 1.0

        date_str = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        global_fallback = daily_means.get("_global_", 1.0)
        
        df["clean_mkt"] = df["market_index"].copy()
        missing_mask = df["clean_mkt"].isna()
        df.loc[missing_mask, "clean_mkt"] = date_str[missing_mask].map(daily_means).fillna(global_fallback)
    return df, daily_means


def fill_coordinates(df: pd.DataFrame, city_coords: Dict[str, tuple[float, float]]) -> pd.DataFrame:
    """
    Ensures pickup and delivery coordinates are populated.
    """
    df = df.copy()
    for prefix in ["pickup", "delivery"]:
        lat_col = f"{prefix}_lat"
        lon_col = f"{prefix}_lon"
        city_col = prefix
        
        if lat_col not in df.columns or df[lat_col].isna().any():
            if lat_col not in df.columns:
                df[lat_col] = np.nan
            if lon_col not in df.columns:
                df[lon_col] = np.nan
                
            for city, (lat, lon) in city_coords.items():
                mask = df[city_col] == city
                df.loc[mask & df[lat_col].isna(), lat_col] = lat
                df.loc[mask & df[lon_col].isna(), lon_col] = lon
                
    return df


def preprocess_data(
    df: pd.DataFrame,
    city_coords: Dict[str, tuple[float, float]],
    equipment_medians: Optional[Dict[str, float]] = None,
    daily_means: Optional[Dict[str, float]] = None
) -> tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Master preprocessing function for freight data.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = fill_coordinates(df, city_coords)
    df, equip_meds = clean_weights(df, equipment_medians)
    df, mkt_means = clean_market_index(df, daily_means)
    return df, equip_meds, mkt_means
