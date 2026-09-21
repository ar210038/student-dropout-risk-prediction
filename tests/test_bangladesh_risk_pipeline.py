from pathlib import Path
import sys

import joblib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bangladesh_risk_pipeline import (  # noqa: E402
    OVERWHELM_QUESTION,
    PRIMARY_CANDIDATE_FEATURES,
    TARGET_QUESTION,
    TIMESTAMP_QUESTION,
    load_and_clean,
    elevated_risk,
    ordinal_value,
    primary_model_frame,
    risk_class,
)
from train_bangladesh_risk_model import (  # noqa: E402
    FINAL_FEATURES,
    LABEL_TO_ID,
    candidate_models,
)


RAW = ROOT / "data" / "bangladesh_risk" / "raw" / "Student Dropout Risk Survey Dataset.xlsx"


def test_duplicate_removal_and_category_normalization():
    cleaned, audit = load_and_clean(RAW)
    assert audit["duplicate_group_count"] == 2
    assert audit["duplicate_group_sizes"] == [16, 3]
    assert audit["duplicate_copies_removed"] == 17
    assert len(cleaned) == 351
    assert set(cleaned["university_type"]) == {
        "Public University",
        "Private University",
        "National University / Affiliated College",
    }


def test_target_conversion_and_risk_mapping():
    assert ordinal_value("1 (Very unlikely)") == 1
    assert ordinal_value(5) == 5
    assert risk_class(1) == "Low Risk"
    assert risk_class(2) == "Low Risk"
    assert risk_class(3) == "Moderate Risk"
    assert risk_class(4) == "High Risk"
    assert risk_class(5) == "High Risk"
    assert [elevated_risk(score) for score in range(1, 6)] == [0, 0, 0, 1, 1]


def test_binary_target_counts_match_cleaned_survey():
    cleaned, _ = load_and_clean(RAW)
    target = cleaned["dropout_intention_score"].map(elevated_risk)
    assert target.value_counts().to_dict() == {0: 274, 1: 77}


def test_primary_schema_excludes_timestamp_target_and_overwhelm():
    cleaned, _ = load_and_clean(RAW)
    frame = primary_model_frame(cleaned, FINAL_FEATURES)
    assert len(FINAL_FEATURES) == 12
    assert set(frame.columns) == {*FINAL_FEATURES, "risk_class"}
    assert TIMESTAMP_QUESTION not in frame.columns
    assert TARGET_QUESTION not in frame.columns
    assert OVERWHELM_QUESTION not in frame.columns
    assert "academic_overwhelm" not in PRIMARY_CANDIDATE_FEATURES


def test_model_round_trip_and_valid_class_predictions(tmp_path):
    cleaned, _ = load_and_clean(RAW)
    model = candidate_models()["Multinomial Logistic Regression"]
    model.fit(cleaned[FINAL_FEATURES], cleaned["risk_class"].map(LABEL_TO_ID))
    model_path = tmp_path / "risk_model.joblib"
    joblib.dump(model, model_path)
    model = joblib.load(model_path)
    predictions = model.predict(cleaned[FINAL_FEATURES].head(5))
    probabilities = model.predict_proba(cleaned[FINAL_FEATURES].head(5))
    assert set(predictions).issubset({0, 1, 2})
    assert probabilities.shape == (5, 3)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
