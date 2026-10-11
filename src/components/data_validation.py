import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging


class DataValidationConfig:
    def __init__(self):
        self.validation_status_file = os.path.join("artifacts", "validation_status.txt")


class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()

    def initiate_data_validation(
        self, train_path, test_path, stores_path, oil_path, holidays_path
    ) -> bool:
        try:
            logging.info("Starting Data Validation component...")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            stores_df = pd.read_csv(stores_path)
            oil_df = pd.read_csv(oil_path)
            holidays_df = pd.read_csv(holidays_path)

            validation_status = True

            # 1. Check for missing critical columns in train dataset
            required_train_cols = [
                "id",
                "date",
                "store_nbr",
                "family",
                "sales",
                "onpromotion",
            ]
            for col in required_train_cols:
                if col not in train_df.columns:
                    validation_status = False
                    logging.error(f"Missing required column in train.csv: {col}")

            # 2. Check for missing columns in test dataset
            required_test_cols = ["id", "date", "store_nbr", "family", "onpromotion"]
            for col in required_test_cols:
                if col not in test_df.columns:
                    validation_status = False
                    logging.error(f"Missing required column in test.csv: {col}")

            # 3. Log null value and shape audit stats
            logging.info(
                f"Dataset audit - Train: {train_df.shape}, Test: {test_df.shape}, "
                f"Stores: {stores_df.shape}, Oil: {oil_df.shape}, Holidays: {holidays_df.shape}"
            )
            train_nulls = train_df.isnull().sum().to_dict()
            logging.info(f"Train dataset null value summary: {train_nulls}")

            os.makedirs("artifacts", exist_ok=True)
            with open(self.data_validation_config.validation_status_file, "w") as f:
                f.write(f"Validation status: {validation_status}")

            logging.info(f"Data Validation completed. Status: {validation_status}")
            return validation_status

        except Exception as e:
            raise CustomException(e, sys)
