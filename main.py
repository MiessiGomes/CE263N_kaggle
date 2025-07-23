import warnings

import numpy as np

from src.config import FEATURES, SUBMISSION_FILE, TARGET, TEST_FILE, TRAIN_FILE
from src.data_processing import clean_data, load_data
from src.feature_engineering import create_features
from src.model_tuner import tune_hyperparameters
from src.predict import make_predictions
from src.train import train_model

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    """Main function to run the entire ML pipeline."""
    print("--- Loading data ---")
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    print("Data loaded successfully.")

    print("\n--- Cleaning data ---")
    train_df_cleaned = clean_data(train_df)
    print(f"Cleaned data shape: {train_df_cleaned.shape}")

    print("\n--- Creating features ---")
    train_df_featured = create_features(train_df_cleaned)
    test_df_featured = create_features(test_df)
    print("Features created successfully.")

    # Prepare Data for Modeling
    X = train_df_featured[FEATURES]
    y = np.log1p(train_df_featured[TARGET])

    print("\n--- Tuning hyperparameters with Optuna ---")
    best_params = tune_hyperparameters(X, y)

    print("\n--- Training final model ---")
    final_model = train_model(X, y, best_params)
    print("Model trained successfully.")

    print("\n--- Making predictions ---")
    submission_df = make_predictions(final_model, test_df_featured)
    print("Predictions made successfully.")

    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission file '{SUBMISSION_FILE}' created successfully.")
    print("\nTop 10 predictions:")
    print(submission_df.head(10))


if __name__ == "__main__":
    main()
