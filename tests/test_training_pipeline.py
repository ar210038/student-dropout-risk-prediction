"""Lightweight checks for the reproducible training pipeline."""

from __future__ import annotations

import joblib
import numpy as np

from src.train_model import (
    DATASET_PATH,
    FEATURES,
    MODEL_PATH,
    build_model_pipelines,
    load_raw_dataset,
    prepare_modeling_data,
)


def test_target_mapping_and_enrolled_exclusion() -> None:
    raw = load_raw_dataset(DATASET_PATH)
    X, y = prepare_modeling_data(raw)

    assert len(X) == len(y) == 3_630
    assert set(y.unique()) == {0, 1}
    assert int((y == 1).sum()) == 1_421
    assert int((y == 0).sum()) == 2_209


def test_expected_feature_columns() -> None:
    raw = load_raw_dataset(DATASET_PATH)
    X, _ = prepare_modeling_data(raw)

    assert list(X.columns) == FEATURES
    assert len(FEATURES) == 24
    assert not any("Curricular units" in feature for feature in FEATURES)


def test_preprocessing_pipeline_accepts_valid_data_and_outputs_probabilities() -> None:
    raw = load_raw_dataset(DATASET_PATH)
    X, y = prepare_modeling_data(raw)
    sample_indices = list(y[y == 0].index[:40]) + list(y[y == 1].index[:40])
    sample_X = X.loc[sample_indices]
    sample_y = y.loc[sample_indices]

    pipeline = build_model_pipelines()["Logistic Regression"]
    pipeline.fit(sample_X, sample_y)
    probabilities = pipeline.predict_proba(sample_X)[:, 1]

    assert probabilities.shape == (len(sample_X),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_saved_model_can_be_loaded_and_predict_probabilities() -> None:
    assert MODEL_PATH.exists(), "Run python src/train_model.py before this test."
    model = joblib.load(MODEL_PATH)
    raw = load_raw_dataset(DATASET_PATH)
    X, _ = prepare_modeling_data(raw)
    probabilities = model.predict_proba(X.head(5))[:, 1]

    assert probabilities.shape == (5,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
