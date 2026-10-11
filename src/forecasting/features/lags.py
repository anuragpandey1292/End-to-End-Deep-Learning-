import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame, lag_days=None, target_col="sales"
) -> pd.DataFrame:
    """
    Computes grouped historical shift/lag features per store and family.
    """
    if lag_days is None:
        lag_days = [7, 14]

    df = df.copy()
    if target_col in df.columns:
        df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        for lag in lag_days:
            col_name = f"{target_col}_lag_{lag}"
            df[col_name] = df.groupby(["store_nbr", "family"])[target_col].shift(lag)
    else:
        for lag in lag_days:
            col_name = f"{target_col}_lag_{lag}"
            df[col_name] = np.nan

    return df
