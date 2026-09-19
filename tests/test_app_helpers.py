"""Focused checks for inference helpers used by the Streamlit interface."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from app import (
    CATEGORY_MAPPINGS,
    METADATA_PATH,
    MODEL_PATH,
    OBSERVED_CODES,
    dropout_probability,
    individual_contributions,
    make_input_frame,
    risk_level,
)
from src.train_model import DATASET_PATH, load_raw_dataset, prepare_modeling_data


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "Low Risk"), (0.299, "Low Risk"), (0.30, "Moderate Risk"),
     (0.399, "Moderate Risk"), (0.40, "High Risk"), (1.0, "High Risk")],
)
def test_risk_bands(probability: float, expected: str) -> None:
    assert risk_level(probability) == expected


def test_all_observed_category_codes_have_readable_labels() -> None:
    assert set(OBSERVED_CODES) == set(CATEGORY_MAPPINGS)
    for feature, codes in OBSERVED_CODES.items():
        assert all(code in CATEGORY_MAPPINGS[feature] for code in codes)


def test_saved_pipeline_accepts_exact_metadata_schema_and_explains_prediction() -> None:
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    pipeline = joblib.load(MODEL_PATH)
    raw = load_raw_dataset(DATASET_PATH)
    features, _ = prepare_modeling_data(raw)
    values = features.iloc[0].to_dict()

    row = make_input_frame(values, metadata["feature_list"])
    probability = dropout_probability(pipeline, row)
    contributions = individual_contributions(pipeline, row, metadata)

    assert isinstance(row, pd.DataFrame)
    assert list(row.columns) == metadata["feature_list"]
    assert 0.0 <= probability <= 1.0
    assert len(contributions) == 5
    assert set(contributions["Feature"]).issubset(metadata["feature_list"])
