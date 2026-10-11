import numpy as np
import pandas as pd


def add_rolling_features(
    df: pd.DataFrame, windows=None, target_col="sales"
) -> pd.DataFrame:
    """
    Computes grouped rolling window statistics and moving averages per store and family.
    Uses shift(1) to avoid target leakage into the current timestamp.
    """
    if windows is None:
        windows = [7, 14]

    df = df.copy()
    if target_col in df.columns:
        df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        for w in windows:
            col_name = f"{target_col}_rolling_mean_{w}"
            df[col_name] = df.groupby(["store_nbr", "family"])[target_col].transform(
                lambda x, win=w: x.shift(1).rolling(window=win).mean()
            )
    else:
        for w in windows:
            col_name = f"{target_col}_rolling_mean_{w}"
            df[col_name] = np.nan

    return df
