"""Shared, leakage-safe data preparation for Bangladesh MICS 2019.

The raw SPSS file is never modified.  This module validates the two attendance
variables, constructs the previous-year-attendee cohort, and exposes the exact
feature schema used by the candidate model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadstat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FS_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "mics_bangladesh_2019"
    / "raw"
    / "Bangladesh MICS6 SPSS Datasets"
    / "fs.sav"
)
HL_DATA_PATH = FS_DATA_PATH.with_name("hl.sav")

RANDOM_STATE = 42
TARGET_NAME = "dropout"

SOURCE_COLUMNS = [
    "FS17",
    "CB7",
    "CB9",
    "CB3",
    "HL4",
    "HH6",
    "HH7",
    "CB10A",
    "CB10B",
    "melevel",
    "windex5",
    "fsdisability",
    "HH52",
    "fsweight",
    "stratum",
    "PSU",
    "HH1",
    "HH2",
    "LN",
]

FINAL_FEATURES = [
    "age",
    "sex",
    "division",
    "area",
    "previous_school_level",
    "previous_grade",
    "mothers_education",
    "wealth_quintile",
    "functional_difficulty",
    "children_5_17_in_household",
]

V2_FINAL_FEATURES = [
    "age",
    "sex",
    "division",
    "area",
    "previous_school_level",
    "previous_grade",
    "mothers_education",
    "fathers_education",
    "wealth_quintile",
    "functional_difficulty",
    "children_5_17_in_household",
]

NUMERIC_FEATURES = ["age", "children_5_17_in_household"]
CATEGORICAL_FEATURES = [feature for feature in FINAL_FEATURES if feature not in NUMERIC_FEATURES]

WEBSITE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "division": "Division",
    "area": "Area",
    "previous_school_level": "Previous School Level",
    "previous_grade": "Previous Grade",
    "mothers_education": "Mother's Education",
    "fathers_education": "Father's Education",
    "wealth_quintile": "Household Economic Group",
    "functional_difficulty": "Functional Difficulty / Disability",
    "children_5_17_in_household": "Children Aged 5–17 in Household",
}

VALUE_MAPS: dict[str, dict[float, str]] = {
    "HL4": {1.0: "Male", 2.0: "Female"},
    "HH6": {1.0: "Urban", 2.0: "Rural"},
    "HH7": {
        10.0: "Barishal",
        20.0: "Chattogram",
        30.0: "Dhaka",
        40.0: "Khulna",
        45.0: "Mymensingh",
        50.0: "Rajshahi",
        55.0: "Rangpur",
        60.0: "Sylhet",
    },
    "CB10A": {
        0.0: "Early childhood education",
        1.0: "Primary",
        2.0: "Secondary",
        3.0: "Secondary / higher secondary",
        4.0: "Higher",
    },
    "CB10B": {
        1.0: "Grade 1",
        2.0: "Grade 2",
        3.0: "Grade 3",
        4.0: "Grade 4",
        5.0: "Grade 5",
        6.0: "Grade 6",
        7.0: "Grade 7",
        8.0: "Grade 8",
        9.0: "Grade 9",
        10.0: "SSC / Dakhil",
        11.0: "Grade 11",
        12.0: "HSC / Alim / Diploma / Polytechnic",
        13.0: "Grade 13",
        14.0: "Grade 14",
        15.0: "BA (Pass) / Fazil",
        16.0: "BA (Hons) / MBBS / BSc Engineering",
        17.0: "MA / MS / MSc and above",
        18.0: "Grade 18",
    },
    "melevel": {
        0.0: "Pre-primary or none",
        1.0: "Primary",
        2.0: "Secondary",
        3.0: "Higher secondary or above",
    },
    "windex5": {
        1.0: "Poorest",
        2.0: "Second",
        3.0: "Middle",
        4.0: "Fourth",
        5.0: "Richest",
    },
    "fsdisability": {
        1.0: "Has functional difficulty",
        2.0: "Has no functional difficulty",
    },
    "felevel": {
        0.0: "Pre-primary or none",
        1.0: "Primary",
        2.0: "Secondary",
        3.0: "Higher secondary or above",
        5.0: "No information",
    },
}


def load_fs_dataset(path: Path = FS_DATA_PATH) -> tuple[pd.DataFrame, Any]:
    """Read the untouched SPSS file and preserve its metadata/value labels."""
    if not path.exists():
        raise FileNotFoundError(f"MICS fs.sav not found: {path}")
    return pyreadstat.read_sav(path, apply_value_formats=False, user_missing=True)


def load_household_roster(path: Path = HL_DATA_PATH) -> tuple[pd.DataFrame, Any]:
    """Load only the roster fields needed for the audited father-education join."""
    if not path.exists():
        raise FileNotFoundError(f"MICS hl.sav not found: {path}")
    columns = ["HH1", "HH2", "HL1", "FLINE", "felevel"]
    return pyreadstat.read_sav(
        path,
        usecols=columns,
        apply_value_formats=False,
        user_missing=True,
    )


def validate_attendance_metadata(metadata: Any) -> None:
    """Fail closed unless CB7 and CB9 have the verified MICS semantics."""
    expected = {1.0: "YES", 2.0: "NO", 9.0: "NO RESPONSE"}
    expected_labels = {
        "CB7": "Attended school or early childhood programme during current school year",
        "CB9": "Attended school or early childhood programme during previous school year",
    }
    for variable, expected_label in expected_labels.items():
        actual_label = metadata.column_names_to_labels.get(variable)
        actual_values = metadata.variable_value_labels.get(variable)
        if actual_label != expected_label:
            raise ValueError(f"Unexpected {variable} variable label: {actual_label!r}")
        if actual_values != expected:
            raise ValueError(f"Unexpected {variable} value labels: {actual_values!r}")


def _map_values(series: pd.Series, variable: str) -> pd.Series:
    mapped = series.map(VALUE_MAPS[variable])
    unexpected = sorted(series[mapped.isna() & series.notna()].unique().tolist())
    # Explicit SPSS nonresponse codes are treated as missing, not as categories.
    allowed_missing_codes = {9.0, 99.0}
    unexpected = [value for value in unexpected if value not in allowed_missing_codes]
    if unexpected:
        raise ValueError(f"Unexpected values in {variable}: {unexpected}")
    return mapped.astype("object")


def construct_dropout_cohort(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, int]]:
    """Construct the completed-interview, previous-year-attendee cohort.

    Target definition:
      * 1 (dropout): CB9=YES and CB7=NO
      * 0 (continued): CB9=YES and CB7=YES
    """
    missing_columns = sorted(set(SOURCE_COLUMNS).difference(raw.columns))
    if missing_columns:
        raise ValueError(f"fs.sav is missing required columns: {missing_columns}")

    completed = raw["FS17"].eq(1.0)
    previous_known = raw["CB9"].isin([1.0, 2.0])
    previous_attended = raw["CB9"].eq(1.0)
    current_known = raw["CB7"].isin([1.0, 2.0])

    audit = {
        "initial_records": int(len(raw)),
        "excluded_incomplete_interview": int((~completed).sum()),
        "excluded_unknown_previous_attendance_after_completed": int(
            (completed & ~previous_known).sum()
        ),
        "excluded_not_attending_previous_year": int(
            (completed & raw["CB9"].eq(2.0)).sum()
        ),
        "excluded_unknown_current_attendance_among_previous_attendees": int(
            (completed & previous_attended & ~current_known).sum()
        ),
    }

    eligible_mask = completed & previous_attended & current_known
    cohort = raw.loc[eligible_mask].copy()
    audit["final_cohort"] = int(len(cohort))

    if cohort.duplicated(["HH1", "HH2", "LN"]).any():
        raise ValueError("Duplicate child keys detected in the final MICS cohort.")

    y = cohort["CB7"].eq(2.0).astype("int8").rename(TARGET_NAME)

    X = pd.DataFrame(index=cohort.index)
    X["age"] = pd.to_numeric(cohort["CB3"], errors="coerce")
    X["sex"] = _map_values(cohort["HL4"], "HL4")
    X["division"] = _map_values(cohort["HH7"], "HH7")
    X["area"] = _map_values(cohort["HH6"], "HH6")
    X["previous_school_level"] = _map_values(cohort["CB10A"], "CB10A")
    X["previous_grade"] = _map_values(cohort["CB10B"], "CB10B")
    ece_without_grade = cohort["CB10A"].eq(0.0) & cohort["CB10B"].isna()
    X.loc[ece_without_grade, "previous_grade"] = "Not applicable (ECE)"
    X["mothers_education"] = _map_values(cohort["melevel"], "melevel")
    X["wealth_quintile"] = _map_values(cohort["windex5"], "windex5")
    X["functional_difficulty"] = _map_values(
        cohort["fsdisability"], "fsdisability"
    )
    X["children_5_17_in_household"] = pd.to_numeric(
        cohort["HH52"], errors="coerce"
    )

    if X[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("Unexpected missing values in numeric MICS features.")
    if not np.isfinite(X[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric MICS feature values detected.")

    context = cohort[
        ["HH1", "HH2", "LN", "fsweight", "stratum", "PSU", "HL4", "HH6", "HH7", "windex5"]
    ].copy()
    return X.loc[:, FINAL_FEATURES], y, context, audit


def construct_dropout_cohort_v2(
    raw: pd.DataFrame,
    household_roster: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict[str, int]]:
    """Extend the validated v1 cohort with a one-to-one father-education join.

    `FS3`/`LN` is the selected child's household-member line number.  It joins
    to `HL1` within the same `HH1`/`HH2` household.  The roster's derived
    `felevel` value is then attached to that child without expanding rows.
    """
    X, y, context, audit = construct_dropout_cohort(raw)
    required = {"HH1", "HH2", "HL1", "FLINE", "felevel"}
    missing = sorted(required.difference(household_roster.columns))
    if missing:
        raise ValueError(f"hl.sav is missing father-education join columns: {missing}")
    if household_roster.duplicated(["HH1", "HH2", "HL1"]).any():
        raise ValueError("Household roster member keys are not unique.")

    child_keys = context[["HH1", "HH2", "LN"]].copy()
    roster = household_roster[["HH1", "HH2", "HL1", "FLINE", "felevel"]].copy()
    merged = child_keys.merge(
        roster,
        left_on=["HH1", "HH2", "LN"],
        right_on=["HH1", "HH2", "HL1"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != len(X) or not merged["_merge"].eq("both").all():
        raise ValueError("Father-education roster merge did not match every child once.")

    father_education = _map_values(merged["felevel"], "felevel")
    father_education.index = X.index
    X = X.copy()
    X["fathers_education"] = father_education
    audit = dict(audit)
    audit.update(
        {
            "father_merge_rows": int(len(merged)),
            "father_merge_unmatched": int((merged["_merge"] != "both").sum()),
            "father_education_no_information": int(merged["felevel"].eq(5.0).sum()),
            "father_education_missing_or_dk": int(merged["felevel"].eq(9.0).sum()),
        }
    )
    return X.loc[:, V2_FINAL_FEATURES], y, context, audit


def make_preprocessor() -> ColumnTransformer:
    """Create fold-safe preprocessing for the two numeric and eight categorical inputs."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_preprocessor_v2() -> ColumnTransformer:
    """Fold-safe preprocessing for the v2 schema including father education."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_features = [
        feature for feature in V2_FINAL_FEATURES if feature not in NUMERIC_FEATURES
    ]
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
    )
