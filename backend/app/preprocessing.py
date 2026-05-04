from __future__ import annotations

from typing import Iterable

import pandas as pd

RAW_NUMERICAL_FEATURES = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
RAW_CATEGORICAL_FEATURES = ["Sex"]
RAW_FEATURES = RAW_CATEGORICAL_FEATURES + RAW_NUMERICAL_FEATURES


def normalize_sex_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def cross_term_names(numerical_features: Iterable[str]) -> list[str]:
    names: list[str] = []
    features = list(numerical_features)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1 = features[i]
            feature2 = features[j]
            names.append(f"{feature1}_x_{feature2}")
    return names


def add_feature_cross_terms(
    df: pd.DataFrame,
    numerical_features: Iterable[str] = RAW_NUMERICAL_FEATURES,
) -> pd.DataFrame:
    df_new = df.copy()
    features = list(numerical_features)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature1 = features[i]
            feature2 = features[j]
            cross_term_name = f"{feature1}_x_{feature2}"
            df_new[cross_term_name] = df_new[feature1] * df_new[feature2]
    return df_new


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_list}")


def prepare_features(
    df: pd.DataFrame,
    label_encoder,
    feature_order: list[str],
) -> pd.DataFrame:
    require_columns(df, RAW_FEATURES)

    df_prepared = df.copy()
    df_prepared["Sex"] = normalize_sex_series(df_prepared["Sex"])
    df_prepared = add_feature_cross_terms(df_prepared, RAW_NUMERICAL_FEATURES)

    try:
        df_prepared["Sex"] = label_encoder.transform(df_prepared["Sex"])
    except ValueError as exc:
        allowed_values = ", ".join(label_encoder.classes_)
        raise ValueError(f"Sex must be one of: {allowed_values}") from exc

    df_prepared["Sex"] = df_prepared["Sex"].astype("category")
    return df_prepared[feature_order]
