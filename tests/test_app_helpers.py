"""Safety and inference checks for the final Streamlit interface."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
from streamlit.testing.v1 import AppTest
from app import (FACTOR_ASSOCIATIONS, INPUT_OPTIONS, METADATA_PATH, MODEL_PATH,
                 PROFILE_FEATURES, PROFILE_NAMES, PROFILE_PATH, assign_profile,
                 make_profile_input)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reports" / "bangladesh_risk" / "clustering_results.json"

def test_exact_profile_schema_excludes_target_and_timestamp():
    assert len(PROFILE_FEATURES) == 12
    forbidden = {"Timestamp", "dropout_intention_score", "academic_overwhelm", "risk_class"}
    assert forbidden.isdisjoint(PROFILE_FEATURES)
    assert set(INPUT_OPTIONS) == set(PROFILE_FEATURES)

def test_saved_profiler_loads_and_profile_mappings_match_metadata():
    profiler = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    assert metadata["features"] == PROFILE_FEATURES
    assert {item["cluster_id"] for item in profiles} == {0, 1}
    assert {item["profile_label"] for item in profiles} == set(PROFILE_NAMES.values())
    assert hasattr(profiler, "predict")

def test_both_profile_ids_and_deterministic_assignment():
    profiler = joblib.load(MODEL_PATH)
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    for profile in profiles:
        answers = {feature: profile["dominant_values"][feature]["value"] for feature in PROFILE_FEATURES}
        answers["internet_quality"] = int(answers["internet_quality"])
        expected = profile["cluster_id"]
        assert assign_profile(profiler, answers) == expected
        assert assign_profile(profiler, answers) == expected

def test_input_frame_column_order():
    values = {feature: INPUT_OPTIONS[feature][0] for feature in PROFILE_FEATURES}
    frame = make_profile_input(values)
    assert list(frame.columns) == PROFILE_FEATURES
    assert len(frame) == 1

def test_profile_statistics_and_scientific_values():
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert [item["size"] for item in profiles] == [213, 138]
    assert [round(item["percentage"], 1) for item in profiles] == [60.7, 39.3]
    assert [round(item["elevated_percentage"], 1) for item in profiles] == [20.7, 23.9]
    assert FACTOR_ASSOCIATIONS["Cramér’s V"].tolist() == [0.185, 0.171, 0.170, 0.169, 0.167, 0.153]
    report_values = {item["feature"]: item["cramers_v"] for item in results["individual_factor_associations"]}
    expected = ["study_material_satisfaction", "scholarship", "study_space_quality", "employment_type", "prior_residence", "sleep_duration"]
    assert [round(report_values[name], 3) for name in expected] == FACTOR_ASSOCIATIONS["Cramér’s V"].tolist()
    risk_test = results["risk_cluster_chi_square"]
    assert round(risk_test["chi_square"], 3) == 0.346
    assert round(risk_test["p_value"], 3) == 0.557
    assert round(risk_test["cramers_v"], 3) == 0.038

def test_active_ui_has_no_legacy_or_individual_forecast_claims():
    source = (ROOT / "app.py").read_text(encoding="utf-8").lower()
    forbidden = ["predict dropout", "dropout probability", "portuguese", "unemployment rate",
                 "inflation rate", "nationality codes", "logistic regression", "oulad", "upv"]
    assert all(phrase not in source for phrase in forbidden)
    assert '"0.557"' in source
    assert '"0.038"' in source

def test_all_four_pages_render_without_exceptions():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    expected_titles = {
        "Home": "Student Profiling and Dropout Risk Factor Analysis",
        "Student Profile Explorer": "Student Profile Explorer",
        "Dropout Risk Factor Analysis": "Dropout Risk Factor Analysis",
        "Methodology & Dataset": "Methodology & Dataset",
    }
    for page, title in expected_titles.items():
        app.radio[0].set_value(page).run()
        assert not app.exception
        assert title in [item.value for item in app.title]

def test_valid_dominant_answers_render_both_profile_names():
    profiles = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    label_to_feature = {label: feature for feature, label in __import__("app").INPUT_LABELS.items()}
    for profile in profiles:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.radio[0].set_value("Student Profile Explorer").run()
        answers = {feature: profile["dominant_values"][feature]["value"] for feature in PROFILE_FEATURES}
        answers["internet_quality"] = int(answers["internet_quality"])
        for widget in app.selectbox:
            widget.set_value(answers[label_to_feature[widget.label]])
        app.select_slider[0].set_value(answers["internet_quality"])
        app.button[0].click().run()
        rendered = " ".join(item.value for item in app.markdown)
        assert not app.exception
        assert profile["profile_label"].split(" — ", maxsplit=1)[1] in rendered
