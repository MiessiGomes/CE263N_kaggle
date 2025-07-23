import numpy as np
import pandas as pd


def haversine_distance(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Calculates the Haversine distance between two points.

    Args:
        lat1: Latitude of the starting point.
        lon1: Longitude of the starting point.
        lat2: Latitude of the ending point.
        lon2: Longitude of the ending point.

    Returns:
        The distance in kilometers.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates time-based and distance features.

    Args:
        df: The DataFrame to add features to.

    Returns:
        The DataFrame with new features.
    """
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year
    df["distance"] = haversine_distance(
        df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
    )
    return df
