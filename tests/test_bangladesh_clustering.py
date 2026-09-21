import json
from pathlib import Path

import joblib
import numpy as np

from src.bangladesh_clustering import (
    CLUSTER_FEATURES,
    FINAL_PROFILE_FEATURES,
    MODEL_PATH,
    PROFILE_PATH,
    RAW_PATH,
    make_preprocessor,
)
from src.bangladesh_risk_pipeline import load_and_clean, primary_model_frame


FORBIDDEN = {"Timestamp", "dropout_intention_score", "risk_class", "academic_overwhelm"}


def test_clustering_features_exclude_target_and_leakage_fields():
    assert FORBIDDEN.isdisjoint(CLUSTER_FEATURES)
    assert FORBIDDEN.isdisjoint(FINAL_PROFILE_FEATURES)
    assert len(FINAL_PROFILE_FEATURES) == 12


def test_preprocessor_schema_is_complete_and_finite():
    cleaned, _ = load_and_clean(RAW_PATH)
    frame = primary_model_frame(cleaned, FINAL_PROFILE_FEATURES)[FINAL_PROFILE_FEATURES]
    matrix = make_preprocessor(FINAL_PROFILE_FEATURES).fit_transform(frame)
    assert matrix.shape[0] == 351
    assert matrix.shape[1] >= len(FINAL_PROFILE_FEATURES)
    assert np.isfinite(matrix).all()


def test_saved_profiler_loads_and_is_deterministic():
    cleaned, _ = load_and_clean(RAW_PATH)
    frame = primary_model_frame(cleaned, FINAL_PROFILE_FEATURES)[FINAL_PROFILE_FEATURES]
    profiler = joblib.load(MODEL_PATH)
    first = profiler.predict(frame)
    second = profiler.predict(frame)
    assert np.array_equal(first, second)
    assert set(first) == {0, 1}


def test_profile_sizes_and_risk_percentages_are_consistent():
    profiles = json.loads(Path(PROFILE_PATH).read_text(encoding="utf-8"))
    assert sum(profile["size"] for profile in profiles) == 351
    assert sum(profile["elevated_count"] for profile in profiles) == 77
    for profile in profiles:
        expected = 100 * profile["elevated_count"] / profile["size"]
        assert profile["elevated_percentage"] == expected
        assert sum(profile["risk_distribution_1_to_5"].values()) == profile["size"]
        assert profile["profile_label"].startswith("Profile ")
