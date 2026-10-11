import pandas as pd


def add_known_covariates(
    df: pd.DataFrame, stores_df: pd.DataFrame = None, oil_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Integrates known deterministic future covariates:
    - Store metadata (city, state, store type, cluster)
    - Oil prices (dcoilwtico, forward & back filled)
    """
    df = df.copy()

    # Store metadata
    if stores_df is not None and not stores_df.empty:
        if "city" not in df.columns or "type" not in df.columns:
            df = df.merge(stores_df, on="store_nbr", how="left")

    # Oil prices
    if oil_df is not None and not oil_df.empty:
        oil_copy = oil_df.copy()
        oil_copy["date"] = pd.to_datetime(oil_copy["date"])
        if "dcoilwtico" not in df.columns:
            df = df.merge(oil_copy, on="date", how="left")
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    elif "dcoilwtico" in df.columns:
        df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    else:
        df["dcoilwtico"] = 0.0

    return df
