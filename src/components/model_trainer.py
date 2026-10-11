import os
import sys
from dataclasses import dataclass
import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pth")

class LSTMModel(nn.Module):
    """Advanced Recurrent Neural Network for Time-Series Forecasting"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Reshape for LSTM: (batch, seq_len=1, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Preparing tensors for multi-model training with MLflow tracking")
            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            input_dim = X_train.shape[1]

            # Start MLflow Experiment Tracking
            mlflow.set_experiment("Store_Sales_Deep_Learning")

            models_to_test = {
                "LSTM_Model": LSTMModel(input_dim),
            }

            best_loss = float('inf')
            best_model_name = ""
            best_model_obj = None

            for model_name, model in models_to_test.items():
                with mlflow.start_run(run_name=model_name):
                    logging.info(f"Training architecture: {model_name}")
                    
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)
                    epochs = 5 # Increase for full training

                    model.train()
                    for epoch in range(epochs):
                        running_loss = 0.0
                        for inputs, targets in train_loader:
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()
                        
                        epoch_loss = running_loss / len(train_loader)
                        logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

                    # Log parameters and metrics to MLflow
                    mlflow.log_param("model_architecture", model_name)
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_metric("final_training_loss", epoch_loss)

                    # Track the best model
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_name = model_name
                        best_model_obj = model

            # Save the winning model weights locally and log to MLflow
            torch.save(best_model_obj.state_dict(), self.model_trainer_config.trained_model_file_path)
            
            try:
                sample_input = X_train_tensor[:1]
                mlflow.pytorch.log_model(
                    pytorch_model=best_model_obj,
                    artifact_path="best_model",
                    input_example=sample_input.numpy()
                )
            except Exception as log_err:
                logging.warning(f"MLflow model logging warning: {log_err}")
            
            logging.info(f"Best Model: {best_model_name} with Loss: {best_loss:.4f}")
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)