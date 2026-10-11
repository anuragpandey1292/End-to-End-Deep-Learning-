import os
import subprocess
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# If dependencies like mlflow are installed in the local .venv, automatically switch to it
venv_python = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", ".venv", "Scripts", "python.exe"
    )
)
if os.path.exists(venv_python) and sys.executable.lower() != venv_python.lower():
    try:
        import mlflow  # noqa: F401
    except ImportError:
        # Re-execute with .venv Python
        res = subprocess.run([venv_python] + sys.argv)
        sys.exit(res.returncode)

from src.components.data_ingestion import DataIngestion  # noqa: E402
from src.components.data_transformation import DataTransformation  # noqa: E402
from src.components.model_trainer import ModelTrainer  # noqa: E402
from src.exception import CustomException  # noqa: E402
from src.logger import logging  # noqa: E402


class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion phase...")
            data_ingestion = DataIngestion()
            (train_data_path, test_data_path, stores_path, oil_path, holidays_path) = (
                data_ingestion.initiate_data_ingestion()
            )
            logging.info("Data ingestion completed.")

            # Step 2: Data Transformation
            logging.info("Starting data transformation phase...")
            data_transformation = DataTransformation()
            train_arr, val_arr, test_scaled, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_data_path,
                    test_data_path,
                    stores_path,
                    oil_path,
                    holidays_path,
                )
            )
            logging.info("Data transformation completed.")

            # Step 3: Model Trainer
            logging.info("Starting model training phase...")
            model_trainer = ModelTrainer()
            model_path = model_trainer.initiate_model_trainer(train_arr, val_arr)
            logging.info(f"Model training completed. Model saved at: {model_path}")

            print(
                f"End-to-End training pipeline executed successfully! Model path: {model_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
