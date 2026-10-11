import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.forecasting.backtesting.walk_forward import chronological_train_val_split

# Import modular features and backtesting modules from src.forecasting
from src.forecasting.features.calendar import (
    add_calendar_features,
    add_historical_shocks,
)
from src.forecasting.features.known import add_known_covariates
from src.forecasting.features.lags import add_lag_features
from src.forecasting.features.promotions import add_promotion_features
from src.forecasting.features.rolling import add_rolling_features
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "store_nbr",
                "onpromotion",
                "is_promoted",
                "dcoilwtico",
                "cluster",
                "is_holiday",
                "is_payday",
                "is_earthquake_period",
                "year",
                "month",
                "day",
                "dayofweek",
                "is_weekend",
                "sales_lag_7",
                "sales_lag_14",
                "sales_rolling_mean_7",
            ]
            categorical_columns = ["family", "city", "state", "type"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def _apply_feature_pipeline(
        self,
        df: pd.DataFrame,
        stores_df: pd.DataFrame,
        oil_df: pd.DataFrame,
        holidays_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Executes modular feature components sequentially.
        """
        logging.info(
            "Applying modular feature pipeline: Known covariates, Calendar, Shocks, Promotions, Lags, Rolling..."
        )
        # 1. Known future covariates (store metadata, oil)
        df = add_known_covariates(df, stores_df=stores_df, oil_df=oil_df)

        # 2. Calendar and holidays
        df = add_calendar_features(df, holidays_df=holidays_df)

        # 3. Historical shocks (earthquake)
        df = add_historical_shocks(df)

        # 4. Promotions
        df = add_promotion_features(df)

        # 5. Grouped lags (7, 14)
        df = add_lag_features(df, lag_days=[7, 14], target_col="sales")

        # 6. Grouped rolling statistics (7-day window)
        df = add_rolling_features(df, windows=[7], target_col="sales")

        return df

    def initiate_data_transformation(
        self, train_path, test_path, stores_path, oil_path, holidays_path
    ):
        try:
            logging.info("Reading datasets for modular chronological transformation...")
            train_df = pd.read_csv(train_path, parse_dates=["date"])
            test_df = pd.read_csv(test_path, parse_dates=["date"])
            stores_df = pd.read_csv(stores_path)
            oil_df = pd.read_csv(oil_path, parse_dates=["date"])
            holidays_df = pd.read_csv(holidays_path, parse_dates=["date"])

            # Apply sequential feature engineering
            train_df = self._apply_feature_pipeline(
                train_df, stores_df, oil_df, holidays_df
            )
            test_df = self._apply_feature_pipeline(
                test_df, stores_df, oil_df, holidays_df
            )

            # --- STRICT CHRONOLOGICAL TIME SPLIT ---
            # Out-of-time walk-forward split using 2017-06-01 cutoff
            sub_train_df, val_df = chronological_train_val_split(
                train_df, cutoff_date="2017-06-01"
            )
            logging.info(
                f"Chronological split completed: Sub-train rows={len(sub_train_df)}, Val rows={len(val_df)}"
            )

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "sales"

            # Features & Targets
            X_train_raw = sub_train_df.drop(
                columns=[target_column_name, "date", "id"], errors="ignore"
            )
            y_train_raw = sub_train_df[target_column_name].values

            X_val_raw = val_df.drop(
                columns=[target_column_name, "date", "id"], errors="ignore"
            )
            y_val_raw = val_df[target_column_name].values

            X_test_raw = test_df.drop(columns=["date", "id"], errors="ignore")

            logging.info(
                "Fitting preprocessor and transforming chronological splits..."
            )
            X_train_scaled = preprocessing_obj.fit_transform(X_train_raw)
            X_val_scaled = preprocessing_obj.transform(X_val_raw)
            X_test_scaled = preprocessing_obj.transform(X_test_raw)

            train_arr = np.c_[X_train_scaled, y_train_raw]
            val_arr = np.c_[X_val_scaled, y_val_raw]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            logging.info("Chronological data transformation completed successfully.")
            return (
                train_arr,
                val_arr,
                X_test_scaled,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
