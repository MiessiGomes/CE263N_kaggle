import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import FEATURES


def make_predictions(model: lgb.LGBMRegressor, test_df: pd.DataFrame) -> pd.DataFrame:
    """Makes predictions on the test set.

    Args:
        model: The trained LightGBM model.
        test_df: The test DataFrame.

    Returns:
        A DataFrame with the predictions.
    """
    X_test = test_df[FEATURES]

    test_predictions_log = model.predict(X_test)

    # Revert predictions to original scale due to log1p transformation
    test_predictions = np.expm1(test_predictions_log)

    # Ensure predictions are non-negative
    test_predictions[test_predictions < 0] = 0

    submission_df = pd.DataFrame(
        {"row_id": test_df["row_id"], "duration": test_predictions}
    )

    return submission_df
