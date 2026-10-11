import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    stores_path: str = os.path.join('artifacts', "stores.csv")
    oil_path: str = os.path.join('artifacts', "oil.csv")
    holidays_path: str = os.path.join('artifacts', "holidays_events.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting advanced data ingestion with auxiliary datasets...")
        try:
            # Paths to your raw data folder
            train_path = os.path.join('data', 'raw', 'train.csv')
            test_path = os.path.join('data', 'raw', 'test.csv')
            stores_path = os.path.join('data', 'raw', 'stores.csv')
            oil_path = os.path.join('data', 'raw', 'oil.csv')
            holidays_path = os.path.join('data', 'raw', 'holidays_events.csv')

            # Load into DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            stores_df = pd.read_csv(stores_path)
            oil_df = pd.read_csv(oil_path)
            holidays_df = pd.read_csv(holidays_path)

            os.makedirs('artifacts', exist_ok=True)

            # Save copies to artifacts
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)
            stores_df.to_csv(self.ingestion_config.stores_path, index=False)
            oil_df.to_csv(self.ingestion_config.oil_path, index=False)
            holidays_df.to_csv(self.ingestion_config.holidays_path, index=False)

            logging.info("Auxiliary datasets ingested and saved to artifacts successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.stores_path,
                self.ingestion_config.oil_path,
                self.ingestion_config.holidays_path
            )
        except Exception as e:
            raise CustomException(e, sys)