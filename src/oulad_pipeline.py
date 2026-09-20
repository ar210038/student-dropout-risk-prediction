"""Leakage-safe feature construction for the OULAD withdrawal experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


KEY = ["code_module", "code_presentation", "id_student"]
PREDICTION_CUTOFF = 30
TARGET = "withdrawn"
FEATURES = [
    "age_group",
    "previous_education",
    "previous_attempts",
    "studied_credits",
    "registration_lead_days",
    "early_assessments_submitted",
    "early_assessments_missed",
    "early_average_score",
    "early_total_clicks",
    "early_active_days",
    "early_unique_resources",
]
CATEGORICAL_FEATURES = ["age_group", "previous_education"]
NUMERIC_FEATURES = [column for column in FEATURES if column not in CATEGORICAL_FEATURES]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read an OULAD CSV, treating its question-mark marker as missing."""
    return pd.read_csv(path, na_values=["?"], **kwargs)


def load_small_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "studentInfo",
        "studentRegistration",
        "assessments",
        "studentAssessment",
        "courses",
        "vle",
    ]
    return {name: read_csv(raw_dir / f"{name}.csv") for name in names}


def validate_table_keys(tables: dict[str, pd.DataFrame]) -> None:
    """Fail early if documented join cardinalities are not true in this copy."""
    if tables["studentInfo"].duplicated(KEY).any():
        raise ValueError("studentInfo is not unique on the student-course key")
    if tables["studentRegistration"].duplicated(KEY).any():
        raise ValueError("studentRegistration is not unique on the student-course key")
    if tables["assessments"].duplicated("id_assessment").any():
        raise ValueError("id_assessment is not unique in assessments")
    if tables["courses"].duplicated(["code_module", "code_presentation"]).any():
        raise ValueError("courses is not unique on module-presentation")


def build_assessment_features(
    assessments: pd.DataFrame,
    student_assessment: pd.DataFrame,
    cutoff: int = PREDICTION_CUTOFF,
) -> pd.DataFrame:
    """Aggregate only assessments due and submissions made by the cutoff."""
    due = assessments.loc[
        assessments["date"].notna() & (assessments["date"] <= cutoff),
        ["code_module", "code_presentation", "id_assessment"],
    ]
    due_counts = (
        due.groupby(["code_module", "code_presentation"])["id_assessment"]
        .nunique()
        .rename("early_assessments_due")
        .reset_index()
    )
    submitted = student_assessment.loc[
        student_assessment["date_submitted"] <= cutoff
    ].merge(due, on="id_assessment", how="inner", validate="many_to_one")
    per_student = (
        submitted.groupby(KEY, as_index=False)
        .agg(
            early_assessments_submitted=("id_assessment", "nunique"),
            early_average_score=("score", "mean"),
        )
    )
    return per_student.merge(
        due_counts, on=["code_module", "code_presentation"], how="right"
    )


def build_vle_features(
    student_vle_path: Path,
    cutoff: int = PREDICTION_CUTOFF,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Aggregate VLE activity at or before the cutoff without loading 10M rows at once."""
    partials: list[pd.DataFrame] = []
    for chunk in read_csv(student_vle_path, chunksize=chunksize):
        early = chunk.loc[chunk["date"] <= cutoff]
        if early.empty:
            continue
        partials.append(
            early.groupby(KEY, as_index=False).agg(
                early_total_clicks=("sum_click", "sum"),
                early_active_days=("date", lambda values: frozenset(values)),
                early_unique_resources=("id_site", lambda values: frozenset(values)),
            )
        )
    combined = pd.concat(partials, ignore_index=True)
    return combined.groupby(KEY, as_index=False).agg(
        early_total_clicks=("early_total_clicks", "sum"),
        early_active_days=("early_active_days", lambda values: len(set().union(*values))),
        early_unique_resources=(
            "early_unique_resources", lambda values: len(set().union(*values))
        ),
    )


def build_modeling_dataset(
    raw_dir: Path,
    cutoff: int = PREDICTION_CUTOFF,
) -> tuple[pd.DataFrame, dict[str, int]]:
    tables = load_small_tables(raw_dir)
    validate_table_keys(tables)
    info = tables["studentInfo"].copy()
    labels = set(info["final_result"].unique())
    expected = {"Withdrawn", "Fail", "Pass", "Distinction"}
    if labels != expected:
        raise ValueError(f"Unexpected final_result labels: {sorted(labels)}")

    base = info.merge(
        tables["studentRegistration"], on=KEY, how="inner", validate="one_to_one"
    )
    base[TARGET] = (base["final_result"] == "Withdrawn").astype(int)
    early_withdrawal = (base[TARGET] == 1) & (
        base["date_unregistration"].notna()
        & (base["date_unregistration"] <= cutoff)
    )
    not_registered = base["date_registration"].isna() | (
        base["date_registration"] > cutoff
    )
    cohort = base.loc[~early_withdrawal & ~not_registered].copy()

    assessment = build_assessment_features(
        tables["assessments"], tables["studentAssessment"], cutoff
    )
    cohort = cohort.merge(assessment, on=KEY, how="left", validate="many_to_one")
    cohort["early_assessments_due"] = cohort.groupby(
        ["code_module", "code_presentation"]
    )["early_assessments_due"].transform("max").fillna(0)
    cohort["early_assessments_submitted"] = cohort[
        "early_assessments_submitted"
    ].fillna(0)
    cohort["early_assessments_missed"] = (
        cohort["early_assessments_due"] - cohort["early_assessments_submitted"]
    ).clip(lower=0)

    vle = build_vle_features(raw_dir / "studentVle.csv", cutoff)
    cohort = cohort.merge(vle, on=KEY, how="left", validate="one_to_one")
    for column in [
        "early_total_clicks",
        "early_active_days",
        "early_unique_resources",
    ]:
        cohort[column] = cohort[column].fillna(0)

    cohort = cohort.rename(
        columns={
            "age_band": "age_group",
            "highest_education": "previous_education",
            "num_of_prev_attempts": "previous_attempts",
            "disability": "disability_status",
        }
    )
    cohort["registration_lead_days"] = -cohort["date_registration"]
    stats = {
        "total_records": len(base),
        "unique_students": base["id_student"].nunique(),
        "early_withdrawals_excluded": int(early_withdrawal.sum()),
        "late_or_missing_registrations_excluded": int((~early_withdrawal & not_registered).sum()),
        "cohort_records": len(cohort),
        "cohort_unique_students": cohort["id_student"].nunique(),
    }
    return cohort[KEY + FEATURES + [TARGET]].copy(), stats


def grouped_holdout(
    dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(
        splitter.split(dataset, dataset[TARGET], groups=dataset["id_student"])
    )
    return dataset.iloc[train_idx].copy(), dataset.iloc[test_idx].copy()
