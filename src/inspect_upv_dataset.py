"""Read-only integrity and structure inspection for the official UPV release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "upv"
RAW_DIR = DATA_DIR / "raw"
ARCHIVES = {
    "dataset_2018_hash.zip": "43fc067c88f09797dacbcd83ee83cc28",
    "dataset_2021_hash.zip": "50cfb2d0f0a07bf3460bc259ffa84064",
    "dataset_2022_hash.zip": "b5f724fde2c33961f07f26e8fa25b4b7",
}


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_year(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        low_memory=False,
        usecols=usecols,
    )


def verify_archives() -> dict[str, str]:
    actual = {name: md5(DATA_DIR / name) for name in ARCHIVES}
    mismatches = {
        name: (ARCHIVES[name], value)
        for name, value in actual.items()
        if value.lower() != ARCHIVES[name]
    }
    if mismatches:
        raise ValueError(f"Official archive checksum mismatch: {mismatches}")
    return actual


def target_gate() -> None:
    """Prevent modeling until an official source maps A/B to outcome meanings."""
    raise RuntimeError(
        "STOP: official UPV sources name abandono_hash and list values A/B, "
        "but do not document which value means dropout."
    )


def inspect() -> dict:
    checksums = verify_archives()
    yearly = {}
    identity_frames = []
    for path in sorted(RAW_DIR.glob("dataset_*_hash.csv")):
        frame = read_year(path)
        year = int(frame["caca"].iloc[0])
        student_year = frame[["dni_hash", "caca", "abandono_hash"]].drop_duplicates()
        identity_frames.append(
            frame[["dni_hash", "tit_hash", "asi_hash", "caca"]].copy()
        )
        yearly[str(year)] = {
            "file": path.name,
            "rows": len(frame),
            "columns": len(frame.columns),
            "unique_students": frame["dni_hash"].nunique(),
            "unique_degrees": frame["tit_hash"].nunique(),
            "unique_courses": frame["asi_hash"].nunique(),
            "student_year_rows": len(student_year),
            "target_values_unmapped": frame["abandono_hash"].value_counts().to_dict(),
            "missing_cells": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_student_course_rows": int(
                frame.duplicated(["dni_hash", "tit_hash", "asi_hash"]).sum()
            ),
            "columns_list": frame.columns.tolist(),
        }
    combined = pd.concat(identity_frames, ignore_index=True)
    student_years = combined[["dni_hash", "caca"]].drop_duplicates()
    return {
        "checksums": checksums,
        "yearly": yearly,
        "total_raw_records": len(combined),
        "unique_students": int(combined["dni_hash"].nunique()),
        "unique_degrees": int(combined["tit_hash"].nunique()),
        "unique_courses": int(combined["asi_hash"].nunique()),
        "student_year_rows": len(student_years),
        "students_in_multiple_released_years": int(
            (student_years.groupby("dni_hash").size() > 1).sum()
        ),
        "target_gate": "FAILED_UNMAPPED_A_B",
    }


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2, default=int))
