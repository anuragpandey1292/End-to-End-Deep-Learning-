import pandas as pd


def add_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and derives promotional features.
    """
    df = df.copy()
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(0).astype(float)
        # Binary promotion indicator
        df["is_promoted"] = (df["onpromotion"] > 0).astype(int)
    else:
        df["onpromotion"] = 0.0
        df["is_promoted"] = 0
    return df
