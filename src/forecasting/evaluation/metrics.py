import numpy as np


def calculate_quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """
    Computes pinball / quantile loss for a given quantile q (e.g. 0.10, 0.50, 0.90).
    """
    errors = y_true - y_pred
    return float(np.mean(np.maximum(q * errors, (q - 1) * errors)))


def calculate_forecast_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Calculates retail production evaluation metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - RMSLE: Root Mean Squared Logarithmic Error (Kaggle store sales evaluation standard)
    - Quantile Losses: P10, P50, P90
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Sales cannot be negative
    y_true_clean = np.maximum(0, y_true)
    y_pred_clean = np.maximum(0, y_pred)

    mae = float(np.mean(np.abs(y_true_clean - y_pred_clean)))
    rmse = float(np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2)))

    # RMSLE: log1p(y_true) vs log1p(y_pred)
    rmsle = float(
        np.sqrt(np.mean((np.log1p(y_true_clean) - np.log1p(y_pred_clean)) ** 2))
    )

    p10_loss = calculate_quantile_loss(y_true_clean, y_pred_clean, q=0.10)
    p50_loss = calculate_quantile_loss(y_true_clean, y_pred_clean, q=0.50)
    p90_loss = calculate_quantile_loss(y_true_clean, y_pred_clean, q=0.90)

    return {
        "mae": mae,
        "rmse": rmse,
        "rmsle": rmsle,
        "p10_loss": p10_loss,
        "p50_loss": p50_loss,
        "p90_loss": p90_loss,
    }
