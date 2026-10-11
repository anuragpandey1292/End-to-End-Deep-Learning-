import pandas as pd


def add_calendar_features(
    df: pd.DataFrame, holidays_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Extracts date parts, holiday flags, and payday cycles.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Standard calendar components
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Payday indicators (typically 15th and end of month)
    df["is_payday"] = ((df["date"].dt.day == 15) | (df["date"].dt.is_month_end)).astype(
        int
    )

    # Holiday processing
    if holidays_df is not None and not holidays_df.empty:
        holidays_copy = holidays_df.copy()
        holidays_copy["date"] = pd.to_datetime(holidays_copy["date"])

        # Filter transferred holidays
        if "transferred" in holidays_copy.columns:
            holidays_copy = holidays_copy[~holidays_copy["transferred"].astype(bool)]

        # Consider national holidays
        national = holidays_copy[holidays_copy["locale"] == "National"][
            ["date"]
        ].drop_duplicates()
        national["is_holiday"] = 1

        df = df.merge(national, on="date", how="left")
        df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    else:
        if "is_holiday" not in df.columns:
            df["is_holiday"] = 0

    return df


def add_historical_shocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags external shocks such as the 2016 Ecuador earthquake (April 16 to May 15, 2016).
    """
    df = df.copy()
    earthquake_start = pd.to_datetime("2016-04-16")
    earthquake_end = pd.to_datetime("2016-05-15")
    df["is_earthquake_period"] = (
        (df["date"] >= earthquake_start) & (df["date"] <= earthquake_end)
    ).astype(int)
    return df
