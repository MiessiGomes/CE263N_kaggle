import pandas as pd


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads training and test data from specified CSV files.

    Args:
        train_path: The file path for the training data.
        test_path: The file path for the test data.

    Returns:
        A tuple containing the training and test DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the training data by removing null values and outliers.

    Args:
        df: The training DataFrame to clean.

    Returns:
        The cleaned DataFrame.
    """
    df.dropna(inplace=True)

    # 2. Remove trips with zero duration
    df = df[df["duration"] > 0]

    # 3. Remove outliers based on the 99th percentile
    upper_limit = df["duration"].quantile(0.99)
    df = df[df["duration"] <= upper_limit]

    return df
