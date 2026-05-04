from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from xgboost import XGBRegressor

MODEL_FILENAME = "xgb_model.json"
ENCODER_FILENAME = "label_encoder.joblib"
METADATA_FILENAME = "metadata.json"


def get_default_artifacts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts"


def save_artifacts(
    model: XGBRegressor,
    label_encoder,
    metadata: dict[str, Any],
    artifacts_dir: Path | None = None,
) -> Path:
    target_dir = Path(artifacts_dir) if artifacts_dir else get_default_artifacts_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(target_dir / MODEL_FILENAME))
    joblib.dump(label_encoder, target_dir / ENCODER_FILENAME)
    (target_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return target_dir


def load_artifacts(
    artifacts_dir: Path | None = None,
) -> tuple[XGBRegressor, Any, dict[str, Any]]:
    source_dir = Path(artifacts_dir) if artifacts_dir else get_default_artifacts_dir()

    model = XGBRegressor()
    model.load_model(str(source_dir / MODEL_FILENAME))
    model.set_params(device="cpu")

    label_encoder = joblib.load(source_dir / ENCODER_FILENAME)
    metadata = json.loads((source_dir / METADATA_FILENAME).read_text(encoding="utf-8"))

    return model, label_encoder, metadata
