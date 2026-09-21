from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mics_pipeline import (  # noqa: E402
    FINAL_FEATURES,
    FS_DATA_PATH,
    V2_FINAL_FEATURES,
    construct_dropout_cohort,
    construct_dropout_cohort_v2,
    load_fs_dataset,
    load_household_roster,
    validate_attendance_metadata,
)


@pytest.fixture(scope="module")
def prepared():
    raw, metadata = load_fs_dataset()
    validate_attendance_metadata(metadata)
    return raw, construct_dropout_cohort(raw)


def test_target_construction_and_exclusions(prepared):
    raw, (X, y, context, audit) = prepared
    assert raw.shape == (40617, 268)
    assert audit["final_cohort"] == 31101
    assert audit["excluded_incomplete_interview"] == 1231
    assert audit["excluded_unknown_previous_attendance_after_completed"] == 1574
    assert audit["excluded_not_attending_previous_year"] == 6711
    assert int(y.sum()) == 966
    assert int((y == 0).sum()) == 30135
    assert len(X) == len(y) == len(context)


def test_feature_schema_and_no_direct_leakage(prepared):
    _, (X, y, _, _) = prepared
    assert list(X.columns) == FINAL_FEATURES
    assert "CB7" not in X.columns
    assert "CB8A" not in X.columns
    assert "CB8B" not in X.columns
    assert "CB9" not in X.columns
    assert set(y.unique()) == {0, 1}
    # One documented CB10A=9 (NO RESPONSE) is retained as missing and handled
    # by the fold-local categorical imputer.
    assert X.isna().sum().to_dict()["previous_school_level"] == 1
    assert int(X.drop(columns=["previous_school_level"]).isna().sum().sum()) == 0


def test_saved_candidate_model_probabilities_when_available(prepared):
    model_path = PROJECT_ROOT / "models" / "mics_bangladesh_dropout_model.joblib"
    if not model_path.exists():
        pytest.skip("Candidate MICS model has not been trained yet")
    _, (X, _, _, _) = prepared
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X.head(25))[:, 1]
    assert probabilities.shape == (25,)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert list(model.feature_names_in_) == FINAL_FEATURES


def test_raw_file_is_preserved():
    assert FS_DATA_PATH.exists()
    assert FS_DATA_PATH.suffix == ".sav"


def test_v2_father_education_merge_is_one_to_one(prepared):
    raw, _ = prepared
    roster, _ = load_household_roster()
    X, y, context, audit = construct_dropout_cohort_v2(raw, roster)
    assert list(X.columns) == V2_FINAL_FEATURES
    assert len(X) == len(y) == len(context) == 31101
    assert audit["father_merge_rows"] == 31101
    assert audit["father_merge_unmatched"] == 0
    assert audit["father_education_no_information"] == 5140
    assert audit["father_education_missing_or_dk"] == 14
    assert X["fathers_education"].isna().sum() == 14


def test_saved_v2_candidate_and_group_split(prepared):
    model_path = PROJECT_ROOT / "models" / "mics_bangladesh_dropout_model_v2.joblib"
    results_path = PROJECT_ROOT / "reports" / "mics_training_results_v2.json"
    if not model_path.exists() or not results_path.exists():
        pytest.skip("Controlled v2 candidate has not been trained yet")

    import json

    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["target_unchanged"] is True
    assert results["grouped_split"]["psu_overlap"] == 0

    raw, _ = prepared
    roster, _ = load_household_roster()
    X, _, _, _ = construct_dropout_cohort_v2(raw, roster)
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X.head(25))[:, 1]
    assert list(model.feature_names_in_) == V2_FINAL_FEATURES
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
