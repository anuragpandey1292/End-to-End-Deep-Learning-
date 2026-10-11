from collections.abc import Generator

import pandas as pd


def chronological_train_val_split(
    df: pd.DataFrame, cutoff_date: str = "2017-06-01", date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into train and validation sets based on a chronological cutoff date,
    guaranteeing no future information leaks into training.
    """
    cutoff = pd.to_datetime(cutoff_date)
    train_df = df[df[date_col] < cutoff].copy()
    val_df = df[df[date_col] >= cutoff].copy()
    return train_df, val_df


class WalkForwardCV:
    """
    Out-of-time walk-forward chronological cross-validation.
    Simulates rolling production deployments across multiple testing horizons.
    """

    def __init__(
        self, n_splits: int = 3, forecast_horizon_days: int = 16, date_col: str = "date"
    ):
        self.n_splits = n_splits
        self.forecast_horizon_days = forecast_horizon_days
        self.date_col = date_col

    def split(
        self, df: pd.DataFrame
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        max_date = df[self.date_col].max()
        for i in reversed(range(self.n_splits)):
            val_end = max_date - pd.Timedelta(days=i * self.forecast_horizon_days)
            val_start = val_end - pd.Timedelta(days=self.forecast_horizon_days)
            train_fold = df[df[self.date_col] < val_start]
            val_fold = df[
                (df[self.date_col] >= val_start) & (df[self.date_col] <= val_end)
            ]
            yield train_fold, val_fold
