from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Sex: str = Field(..., description="Biological sex: male or female")
    Age: float = Field(..., ge=0)
    Height: float = Field(..., ge=0)
    Weight: float = Field(..., ge=0)
    Duration: float = Field(..., ge=0)
    Heart_Rate: float = Field(..., ge=0)
    Body_Temp: float = Field(..., ge=0)

    @field_validator("Sex")
    @classmethod
    def normalize_sex(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Sex must be a string")
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Sex must be a non-empty string")
        return normalized


class PredictionResponse(BaseModel):
    calories: float = Field(..., ge=0)


class FeatureRange(BaseModel):
    min: float
    max: float


class MetadataResponse(BaseModel):
    feature_names: list[str]
    raw_features: list[str]
    cross_term_features: list[str]
    feature_ranges: dict[str, FeatureRange]
    sex_classes: list[str]
    clip_min: float
    clip_max: float
    target_transform: str
    validation_fraction: float
