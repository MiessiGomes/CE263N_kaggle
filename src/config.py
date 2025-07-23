TRAIN_FILE = "dataset/train.csv"
TEST_FILE = "dataset/test.csv"
SUBMISSION_FILE = "submission_optuna_2.csv"

# Feature lists
FEATURES = [
    "start_lng",
    "start_lat",
    "end_lng",
    "end_lat",
    "hour",
    "day_of_week",
    "month",
    "year",
    "distance",
]

TARGET = "duration"

# Optuna settings
N_TRIALS = 20
