import os
import sys

import pandas as pd
import torch
import torch.nn as nn

from src.components.models.gru import GRUModel
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pth")
            preprocessor_path = os.path.join("artifacts", "proprocessor.pkl")

            logging.info("Loading preprocessor and model for prediction...")

            preprocessor = load_object(file_path=preprocessor_path)

            # Transform incoming features using the saved preprocessor
            data_scaled = preprocessor.transform(features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            # Convert to tensor
            X_tensor = torch.tensor(data_scaled, dtype=torch.float32)

            # Determine input dimension dynamically based on scaled features
            input_dim = X_tensor.shape[1]

            model = GRUModel(input_dim=input_dim, hidden_dim=64)
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device("cpu"))
            )
            model.eval()

            with torch.no_grad():
                predictions = model(X_tensor)

            return predictions.numpy().flatten()

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    This class maps incoming web/API request data into a Pandas DataFrame
    matching your training features structure.
    """

    def __init__(self, store_nbr: int, family: str, onpromotion: int):
        self.store_nbr = store_nbr
        self.family = family
        self.onpromotion = onpromotion

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "store_nbr": [self.store_nbr],
                "family": [self.family],
                "onpromotion": [self.onpromotion],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
