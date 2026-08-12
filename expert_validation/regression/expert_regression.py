#!/usr/bin/env python3

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


TRAINING_FILE = Path("expert_vs_volunteer.csv")
TARGET_FILE = Path("tournament_filter_no_shot.csv")

FEATURE_COLUMNS = [
    "volunteer_rating",
    "AnomalyScore",
]

TARGET_COLUMN = "expert selection percentage"


def main() -> None:
    # ------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------
    if not TRAINING_FILE.is_file():
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE}")

    training_data = pd.read_csv(TRAINING_FILE)

    required_training_columns = {
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing = required_training_columns - set(training_data.columns)

    if missing:
        raise ValueError(
            "Training file is missing columns: "
            + ", ".join(sorted(missing))
        )

    training_data = training_data[
        [*FEATURE_COLUMNS, TARGET_COLUMN]
    ].copy()

    for column in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        training_data[column] = pd.to_numeric(
            training_data[column],
            errors="coerce",
        )

    # Rows without expert scores cannot be used for training.
    training_data = training_data.dropna(subset=[TARGET_COLUMN])

    if training_data.empty:
        raise ValueError("No valid training rows were found.")

    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data[TARGET_COLUMN]

    # ------------------------------------------------------------
    # Fit linear regression
    # ------------------------------------------------------------
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("regression", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)

    regression = model.named_steps["regression"]

    print("Fitted linear regression:")
    print(f"  Intercept: {regression.intercept_:.6f}")

    for feature, coefficient in zip(
        FEATURE_COLUMNS,
        regression.coef_,
    ):
        print(f"  {feature}: {coefficient:.6f}")

    # ------------------------------------------------------------
    # Load the file to which predictions will be appended
    # ------------------------------------------------------------
    if not TARGET_FILE.is_file():
        raise FileNotFoundError(f"Target file not found: {TARGET_FILE}")

    target_data = pd.read_csv(TARGET_FILE)

    missing = set(FEATURE_COLUMNS) - set(target_data.columns)

    if missing:
        raise ValueError(
            "Target file is missing columns: "
            + ", ".join(sorted(missing))
        )

    for column in FEATURE_COLUMNS:
        target_data[column] = pd.to_numeric(
            target_data[column],
            errors="coerce",
        )

    X_new = target_data[FEATURE_COLUMNS]

    # ------------------------------------------------------------
    # Predict expert scores
    # ------------------------------------------------------------
    predictions = model.predict(X_new)

    # Detect whether the training target is on a 0-1 or 0-100 scale.
    if y_train.max() <= 1.0:
        lower_bound = 0.0
        upper_bound = 1.0
    else:
        lower_bound = 0.0
        upper_bound = 100.0

    predictions = np.clip(
        predictions,
        lower_bound,
        upper_bound,
    )

    target_data["expert_score"] = predictions

    # ------------------------------------------------------------
    # Back up the original and overwrite the target CSV
    # ------------------------------------------------------------
    target_data.to_csv(TARGET_FILE, index=False)

    print(f"\nPredicted expert scores for {len(target_data)} rows.")
    print(f"Updated file: {TARGET_FILE.resolve()}")

    print("\nPrediction summary:")
    print(target_data["expert_score"].describe().to_string())


if __name__ == "__main__":
    main()