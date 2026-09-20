"""Active Semester-1 website safety and inference tests."""
import json
from pathlib import Path
import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest
from app import (FORM_FIELDS,METADATA_PATH,MODEL_PATH,OCCUPATIONS,OCCUPATION_BY_LABEL,
    RESULTS_PATH,THRESHOLD,dropout_probability,make_input_frame,risk_label,validate_inputs)

ROOT=Path(__file__).resolve().parents[1]

def sample_values(**overrides):
    values={'Age at enrollment':19,'Previous qualification (grade)':130.0,'Admission grade':130.0,"Mother's occupation":9,"Father's occupation":9,'Scholarship holder':0,'Debtor':0,'Tuition fees up to date':1,'Curricular units 1st sem (enrolled)':6,'Curricular units 1st sem (evaluations)':8,'Curricular units 1st sem (approved)':5,'Curricular units 1st sem (grade)':12.0}
    values.update(overrides); return values

def test_exactly_twelve_fields_and_no_banned_inputs():
    assert len(FORM_FIELDS)==12
    banned=['2nd sem','Application mode','Application order','Course','Nacionality','Unemployment rate','Inflation rate','GDP',"Mother's qualification","Father's qualification"]
    assert all(not any(term.lower() in field.lower() for term in banned) for field in FORM_FIELDS)

def test_occupation_labels_round_trip_without_raw_display_codes():
    assert OCCUPATIONS[4]=='Administrative staff'
    assert OCCUPATION_BY_LABEL['Skilled industry, construction and craft workers']==7
    assert len(OCCUPATIONS)==len(OCCUPATION_BY_LABEL)

def test_courses_passed_validation_and_input_order():
    assert validate_inputs(sample_values(**{'Curricular units 1st sem (enrolled)':4,'Curricular units 1st sem (approved)':5}))==['Courses Passed cannot be greater than Courses Enrolled.']
    assert validate_inputs(sample_values(**{'Curricular units 1st sem (enrolled)':0,'Curricular units 1st sem (approved)':0}))==[]
    assert list(make_input_frame(sample_values()).columns)==FORM_FIELDS

def test_saved_artifact_probability_and_locked_threshold():
    artifact=joblib.load(MODEL_PATH); metadata=json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    probability=dropout_probability(artifact,sample_values())
    assert 0<=probability<=1
    assert artifact['threshold']==metadata['threshold']==THRESHOLD==0.48

def test_both_result_paths():
    assert risk_label(0.479999)=='Lower Estimated Risk'
    assert risk_label(0.48)=='Elevated Estimated Risk'
    artifact=joblib.load(MODEL_PATH)
    raw=pd.read_csv(ROOT/'data'/'student_dropout.csv',sep=';'); raw.columns=raw.columns.str.strip()
    probabilities=artifact['pipeline'].predict_proba(raw[FORM_FIELDS])[:,list(artifact['pipeline'].classes_).index(1)]
    assert risk_label(float(probabilities.min()))=='Lower Estimated Risk'
    assert risk_label(float(probabilities.max()))=='Elevated Estimated Risk'

def test_metrics_match_locked_results():
    r=json.loads(RESULTS_PATH.read_text(encoding='utf-8'))['test_metrics']
    assert round(r['precision'],4)==0.8350 and round(r['recall'],4)==0.8732
    assert round(r['f1'],4)==0.8537 and round(r['roc_auc'],4)==0.9385
    assert round(r['average_precision'],4)==0.9300 and round(r['brier_score'],4)==0.0967

def test_active_source_has_no_old_profile_ui_or_deterministic_claim():
    source=(ROOT/'app.py').read_text(encoding='utf-8').lower()
    banned=['k-means','student profile explorer','profile 1','profile 2','silhouette','davies-bouldin','clustering pca','will drop out']
    assert all(term not in source for term in banned)

def test_all_pages_render_without_exceptions_and_form_has_12_widgets():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    expected={'Home':'Student Dropout Early Warning System','Assess Dropout Risk':'Assess Dropout Risk','Model Performance':'Model Performance','About & Methodology':'About & Methodology'}
    for page,title in expected.items():
        app.radio[0].set_value(page).run(); assert not app.exception; assert title in [item.value for item in app.title]
    app.radio[0].set_value('Assess Dropout Risk').run()
    assert len(app.number_input)+len(app.selectbox)==12
