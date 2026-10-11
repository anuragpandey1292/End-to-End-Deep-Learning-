import os
import sys
from dataclasses import dataclass

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.components.models import CNN1DModel, GRUModel, LSTMModel, MLPModel
from src.exception import CustomException
from src.forecasting.evaluation.metrics import calculate_forecast_metrics
from src.forecasting.tracking.mlflow_tracker import MLflowTracker
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pth")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.tracker = MLflowTracker("Store_Sales_Deep_Learning_Chronological")

    def initiate_model_trainer(self, train_array, val_array):
        try:
            logging.info(
                "Preparing tensors for time-series backtesting and metric evaluation..."
            )

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val, y_val = val_array[:, :-1], val_array[:, -1]

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

            input_dim = X_train.shape[1]

            best_val_rmsle = float("inf")
            best_model_name = ""
            best_model_obj = None

            model_classes = {
                "MLP_Model": MLPModel,
                "CNN_1D_Model": CNN1DModel,
                "LSTM_Model": LSTMModel,
                "GRU_Model": GRUModel,
            }

            for model_name, model_class in model_classes.items():
                logging.info(f"Optimizing architecture: {model_name} with Optuna...")

                def create_objective(m_name, m_cls):
                    def objective(trial):
                        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                        hidden_dim = trial.suggest_categorical(
                            "hidden_dim", [32, 64, 128]
                        )
                        epochs = 3

                        model = m_cls(input_dim=input_dim, hidden_dim=hidden_dim)
                        criterion = nn.MSELoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                        model.train()
                        for _epoch in range(epochs):
                            for inputs, targets in train_loader:
                                optimizer.zero_grad()
                                loss = criterion(model(inputs), targets)
                                loss.backward()
                                optimizer.step()

                        # Evaluate on chronological validation set
                        model.eval()
                        with torch.no_grad():
                            val_preds = model(X_val_tensor).numpy().flatten()

                        metrics = calculate_forecast_metrics(y_val, val_preds)

                        with self.tracker.start_run(
                            run_name=f"{m_name}_trial_{trial.number}", nested=True
                        ):
                            self.tracker.log_params(
                                {
                                    "model_architecture": m_name,
                                    "learning_rate": lr,
                                    "hidden_dim": hidden_dim,
                                }
                            )
                            self.tracker.log_metrics(
                                {
                                    "val_mae": metrics["mae"],
                                    "val_rmse": metrics["rmse"],
                                    "val_rmsle": metrics["rmsle"],
                                    "val_p10_loss": metrics["p10_loss"],
                                    "val_p50_loss": metrics["p50_loss"],
                                    "val_p90_loss": metrics["p90_loss"],
                                }
                            )

                        return metrics["rmsle"]

                    return objective

                study = optuna.create_study(direction="minimize")
                study.optimize(create_objective(model_name, model_class), n_trials=2)

                logging.info(
                    f"Best Validation RMSLE for {model_name}: {study.best_value:.4f}"
                )

                if study.best_value < best_val_rmsle:
                    best_val_rmsle = study.best_value
                    best_model_name = model_name
                    best_params = study.best_params

                    best_model_obj = model_class(
                        input_dim=input_dim, hidden_dim=best_params["hidden_dim"]
                    )
                    optimizer = optim.Adam(
                        best_model_obj.parameters(), lr=best_params["lr"]
                    )
                    criterion = nn.MSELoss()

                    best_model_obj.train()
                    for inputs, targets in train_loader:
                        optimizer.zero_grad()
                        loss = criterion(best_model_obj(inputs), targets)
                        loss.backward()
                        optimizer.step()

            # Save winning model weights
            torch.save(
                best_model_obj.state_dict(),
                self.model_trainer_config.trained_model_file_path,
            )
            self.tracker.log_artifact(self.model_trainer_config.trained_model_file_path)

            logging.info(
                f"Global Winner: {best_model_name} with Validation RMSLE: {best_val_rmsle:.4f}"
            )
            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
