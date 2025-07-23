import lightgbm as lgb
import numpy as np


def train_model(X: np.ndarray, y: np.ndarray, params: dict) -> lgb.LGBMRegressor:
    """Trains the final LightGBM model.

    Args:
        X: The full feature set.
        y: The full target set.
        params: A dictionary with the model's hyperparameters.

    Returns:
        The trained LightGBM model.
    """
    params.update(
        {
            "objective": "regression_l1",
            "metric": "rmse",
            "n_estimators": 300,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
            "boosting_type": "gbdt",
        }
    )

    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)

    return model
