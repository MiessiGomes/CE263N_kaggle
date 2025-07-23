import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.config import N_TRIALS


def objective(
    trial: optuna.trial.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """The objective function for Optuna optimization.

    Args:
        trial: An Optuna trial object.
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        The root mean squared error for the trial.
    """
    params = {
        "objective": "regression_l1",
        "metric": "rmse",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
        "boosting_type": "gbdt",
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))


def tune_hyperparameters(X: np.ndarray, y: np.ndarray) -> dict:
    """Tunes hyperparameters using Optuna.

    Args:
        X: The full feature set.
        y: The full target set.

    Returns:
        A dictionary with the best hyperparameters found.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print(f"Best RMSE (log): {study.best_value:.4f}")
    print("Best parameters found: ", study.best_params)

    return study.best_params
