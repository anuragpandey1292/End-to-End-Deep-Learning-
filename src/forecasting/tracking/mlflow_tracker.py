from typing import Any

import mlflow
import mlflow.pytorch


class MLflowTracker:
    """
    Encapsulates MLflow experiment and run management, logging params, metrics, and models.
    """

    def __init__(
        self, experiment_name: str = "Store_Sales_Deep_Learning_Chronological"
    ):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str | None = None, nested: bool = False):
        return mlflow.start_run(run_name=run_name, nested=nested)

    def log_params(self, params: dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_pytorch_model(
        self, model, artifact_path: str = "best_model", input_example=None
    ):
        try:
            if input_example is not None:
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=artifact_path,
                    input_example=input_example,
                )
            else:
                mlflow.pytorch.log_model(
                    pytorch_model=model, artifact_path=artifact_path
                )
        except Exception:
            # Fall back to logging as artifact if pt2 tracing fails
            pass
