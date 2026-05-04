from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from app.artifacts import save_artifacts
from app.preprocessing import (
    RAW_FEATURES,
    RAW_NUMERICAL_FEATURES,
    add_feature_cross_terms,
    cross_term_names,
    normalize_sex_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save calorie model artifacts.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("train.csv"),
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory to write model artifacts.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="XGBoost device setting.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of data reserved for validation. Use 0 to disable.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data splits.",
    )
    return parser.parse_args()


def build_model(device: str, random_state: int) -> XGBRegressor:
    return XGBRegressor(
        device=device,
        max_depth=10,
        colsample_bytree=0.75,
        subsample=0.9,
        n_estimators=2000,
        learning_rate=0.02,
        gamma=0.01,
        max_delta_step=2,
        eval_metric="rmse",
        enable_categorical=True,
        random_state=random_state,
    )


def main() -> None:
    args = parse_args()
    if not args.train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")

    train_df = pd.read_csv(args.train_csv)
    train_df["Sex"] = normalize_sex_series(train_df["Sex"])
    feature_ranges = {
        feature: {
            "min": float(train_df[feature].min()),
            "max": float(train_df[feature].max()),
        }
        for feature in RAW_NUMERICAL_FEATURES
    }
    clip_min = float(train_df["Calories"].min())
    clip_max = float(train_df["Calories"].max())
    train_df = add_feature_cross_terms(train_df, RAW_NUMERICAL_FEATURES)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Sex"])

    train_df["Sex"] = label_encoder.transform(train_df["Sex"])
    train_df["Sex"] = train_df["Sex"].astype("category")

    X = train_df.drop(columns=["id", "Calories"])
    y = np.log1p(train_df["Calories"])

    model = build_model(args.device, args.random_state)

    validation_fraction = max(0.0, min(args.validation_fraction, 0.9))
    eval_set = None
    if validation_fraction > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=validation_fraction,
            random_state=args.random_state,
        )
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=100,
            early_stopping_rounds=100,
        )
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Validation RMSE: {rmse:.4f}")
    else:
        model.fit(X, y, verbose=False)

    metadata = {
        "feature_names": X.columns.tolist(),
        "raw_features": RAW_FEATURES,
        "cross_term_features": cross_term_names(RAW_NUMERICAL_FEATURES),
        "feature_ranges": feature_ranges,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "target_transform": "log1p",
        "sex_classes": label_encoder.classes_.tolist(),
        "validation_fraction": validation_fraction,
    }

    target_dir = save_artifacts(model, label_encoder, metadata, args.artifacts_dir)
    print(f"Saved artifacts to: {target_dir}")


if __name__ == "__main__":
    main()
