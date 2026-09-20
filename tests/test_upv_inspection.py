from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inspect_upv_dataset import inspect, target_gate, verify_archives  # noqa: E402


def test_official_upv_archive_checksums():
    assert len(verify_archives()) == 3


def test_upv_release_structure_and_identifiers():
    result = inspect()
    assert result["total_raw_records"] == 464_739
    assert result["unique_students"] == 39_364
    assert result["unique_degrees"] == 163
    assert result["unique_courses"] == 4_989
    assert result["student_year_rows"] == 60_924


def test_target_gate_rejects_undocumented_class_mapping():
    with pytest.raises(RuntimeError, match="do not document which value means dropout"):
        target_gate()
