from pathlib import Path
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oulad_pipeline import (  # noqa: E402
    FEATURES,
    TARGET,
    build_assessment_features,
    build_modeling_dataset,
    grouped_holdout,
    load_small_tables,
    validate_table_keys,
)


RAW = ROOT / "data" / "oulad" / "raw"


def test_documented_join_keys_are_unique():
    validate_table_keys(load_small_tables(RAW))


def test_assessment_cutoff_excludes_late_due_and_late_submission():
    assessments = pd.DataFrame(
        {
            "code_module": ["A", "A"],
            "code_presentation": ["1", "1"],
            "id_assessment": [10, 11],
            "date": [20, 40],
        }
    )
    submissions = pd.DataFrame(
        {
            "id_assessment": [10, 10, 11],
            "id_student": [1, 2, 1],
            "date_submitted": [25, 35, 20],
            "score": [80, 70, 99],
        }
    )
    result = build_assessment_features(assessments, submissions, cutoff=30)
    assert result.loc[result["id_student"] == 1, "early_average_score"].item() == 80
    assert 2 not in result["id_student"].dropna().tolist()


def test_target_cutoff_early_withdrawals_and_feature_schema():
    data, stats = build_modeling_dataset(RAW)
    assert set(data[TARGET].unique()) == {0, 1}
    assert stats["early_withdrawals_excluded"] > 0
    assert list(data.columns) == [
        "code_module",
        "code_presentation",
        "id_student",
        *FEATURES,
        TARGET,
    ]
    assert not data.duplicated(["code_module", "code_presentation", "id_student"]).any()


def test_grouped_holdout_has_no_student_overlap():
    data, _ = build_modeling_dataset(RAW)
    train, test = grouped_holdout(data)
    assert set(train["id_student"]).isdisjoint(test["id_student"])


def test_saved_candidate_model_predicts_when_available():
    model_path = ROOT / "models" / "oulad" / "oulad_dropout_model.joblib"
    if not model_path.exists():
        return
    data, _ = build_modeling_dataset(RAW)
    model = joblib.load(model_path)
    probabilities = model.predict_proba(data[FEATURES].head(3))[:, 1]
    assert len(probabilities) == 3
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
