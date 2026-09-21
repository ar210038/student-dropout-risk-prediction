"""Cleaning and modeling constants for the Bangladesh risk survey."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TIMESTAMP_QUESTION = "Timestamp"
TARGET_QUESTION = (
    "How likely are you to consider dropping out due to academic stress? "
    "(Just let your feelings out)"
)
OVERWHELM_QUESTION = "How often do you feel overwhelmed by academic workload?"

QUESTION_TO_FEATURE = {
    "Age:  ": "age_group",
    "Gender:": "gender",
    "Current academic level:": "academic_level",
    "Type of university:": "university_type",
    "Where did you reside before joining university?": "prior_residence",
    "Do you stay in university residence (halls) or at home?": "living_arrangement",
    "Average sleep duration per night (weekdays):": "sleep_duration",
    "Do you work a part-time job or freelance?": "employment_type",
    "How often do you skip meals due to academic/work commitments?": "meal_skipping",
    "Commute/travel time to university (one way):": "commute_time",
    "Rate the quality of your internet access for studying:": "internet_quality",
    "Do you have a dedicated study space at your residence, and if yes, how noisy is it?": "study_space_quality",
    "Current GPA (or equivalent):": "current_gpa",
    "When do you typically study for courses?": "study_routine",
    "Attendance rate in the past month:": "attendance",
    OVERWHELM_QUESTION: "academic_overwhelm",
    "Approximate monthly family income (BDT):": "family_income",
    "Do you receive additional scholarships/stipends?": "scholarship",
    "Personal monthly income (if any):": "personal_income",
    "Do you have access to academic resources (e.g., library, online journals)?": "academic_resource_access",
    "How satisfied are you with university-provided study materials?": "study_material_satisfaction",
    TARGET_QUESTION: "dropout_intention_score",
}

PRIMARY_CANDIDATE_FEATURES = [
    value
    for value in QUESTION_TO_FEATURE.values()
    if value not in {"dropout_intention_score", "academic_overwhelm"}
]
NUMERIC_RATING_FEATURES = [
    "internet_quality",
    "study_material_satisfaction",
]
CATEGORICAL_FEATURES = [
    feature for feature in PRIMARY_CANDIDATE_FEATURES if feature not in NUMERIC_RATING_FEATURES
]
RISK_ORDER = ["Low Risk", "Moderate Risk", "High Risk"]

UNIVERSITY_TYPE_MAP = {
    "Public University (e.g., DU, JU, JNU, BUP)": "Public University",
    "Public University (e.g., Dhaka University, Jahangirnagar University)": "Public University",
    "Private University (e.g., BRAC University, North South University)": "Private University",
    "National University (Affiliated Colleges)": "National University / Affiliated College",
}


def ordinal_value(value: object) -> int:
    """Extract the explicitly stated leading 1–5 rating."""
    if pd.isna(value):
        raise ValueError("Missing ordinal response")
    text = str(value)
    match = re.match(r"\s*([1-5])(?:\.0)?(?:\s|\(|$)", text)
    if not match:
        match = re.search(r"\(([1-5])\)\s*$", text)
    if not match:
        raise ValueError(f"Cannot parse 1–5 response: {value!r}")
    return int(match.group(1))


def risk_class(score: int) -> str:
    if score in (1, 2):
        return "Low Risk"
    if score == 3:
        return "Moderate Risk"
    if score in (4, 5):
        return "High Risk"
    raise ValueError(f"Risk score must be 1–5, got {score}")


def elevated_risk(score: int) -> int:
    """Map self-reported scores 4–5 to the binary elevated-risk class."""
    if score not in (1, 2, 3, 4, 5):
        raise ValueError(f"Risk score must be 1–5, got {score}")
    return int(score >= 4)


def _clean_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    return value.strip().replace("�", "–")


def duplicate_audit(raw: pd.DataFrame) -> dict:
    profile_columns = [column for column in raw.columns if column != TIMESTAMP_QUESTION]
    grouped = raw.groupby(profile_columns, dropna=False, sort=False)
    groups = []
    for _, frame in grouped:
        if len(frame) <= 1:
            continue
        timestamps = pd.to_datetime(frame[TIMESTAMP_QUESTION]).sort_values()
        groups.append(
            {
                "size": len(frame),
                "timestamps": [timestamp.isoformat() for timestamp in timestamps],
                "spacing_seconds": timestamps.diff().dropna().dt.total_seconds().tolist(),
            }
        )
    groups.sort(key=lambda item: item["size"], reverse=True)
    return {
        "duplicate_group_count": len(groups),
        "duplicate_group_sizes": [item["size"] for item in groups],
        "duplicate_copies_removed": sum(item["size"] - 1 for item in groups),
        "groups": groups,
    }


def clean_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    expected = {TIMESTAMP_QUESTION, *QUESTION_TO_FEATURE.keys()}
    if set(raw.columns) != expected or len(raw.columns) != 23:
        raise ValueError("Workbook schema does not match the verified 23-column survey")

    audit = duplicate_audit(raw)
    profile_columns = [column for column in raw.columns if column != TIMESTAMP_QUESTION]
    cleaned = raw.drop_duplicates(subset=profile_columns, keep="first").copy()
    cleaned = cleaned.map(_clean_text)
    cleaned = cleaned.rename(columns=QUESTION_TO_FEATURE)
    cleaned["university_type"] = cleaned["university_type"].replace(UNIVERSITY_TYPE_MAP)
    for feature in NUMERIC_RATING_FEATURES + ["dropout_intention_score"]:
        cleaned[feature] = cleaned[feature].map(ordinal_value)
    cleaned["risk_class"] = cleaned["dropout_intention_score"].map(risk_class)
    cleaned[TIMESTAMP_QUESTION] = pd.to_datetime(cleaned[TIMESTAMP_QUESTION])

    audit.update(
        {
            "records_before": len(raw),
            "records_after": len(cleaned),
            "category_mappings": {
                "university_type": UNIVERSITY_TYPE_MAP,
                "mixed_1_to_5_responses": (
                    "Numeric and annotated values are mapped to their leading integer."
                ),
                "encoding_cleanup": "Replacement character in ranges normalized to an en dash.",
            },
        }
    )
    return cleaned, audit


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    return clean_dataframe(pd.read_excel(path, sheet_name="Sheet1"))


def primary_model_frame(cleaned: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    forbidden = {TIMESTAMP_QUESTION, "dropout_intention_score", "academic_overwhelm"}
    if forbidden.intersection(features):
        raise ValueError("Primary feature set contains a forbidden leakage/overlap field")
    missing = set(features) - set(PRIMARY_CANDIDATE_FEATURES)
    if missing:
        raise ValueError(f"Unknown primary features: {sorted(missing)}")
    return cleaned[features + ["risk_class"]].copy()
