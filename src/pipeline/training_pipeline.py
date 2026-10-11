import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion phase...")
            data_ingestion = DataIngestion()
            (
                train_data_path, 
                test_data_path, 
                stores_path, 
                oil_path, 
                holidays_path
            ) = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")

            # Step 2: Data Transformation
            logging.info("Starting data transformation phase...")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            logging.info("Data transformation completed.")

            # Step 3: Model Trainer
            logging.info("Starting model training phase...")
            model_trainer = ModelTrainer()
            model_path = model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed. Model saved at: {model_path}")

            print(f"End-to-End training pipeline executed successfully! Model path: {model_path}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()