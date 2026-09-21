"""Active Semester-1 website safety and inference tests."""
import json
import hashlib
from pathlib import Path
import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest
from app import (FORM_FIELDS,METADATA_PATH,MODEL_PATH,OCCUPATIONS,OCCUPATION_BY_LABEL,
    THRESHOLD,dropout_probability,make_input_frame,risk_label,
    semester_gpa_to_source_grade,support_guidance,validate_inputs)
from src.uci_sem1_features import normalize_previous_grade

ROOT=Path(__file__).resolve().parents[1]
RESULTS_PATH=ROOT/'reports'/'uci_sem1'/'training_results_11_input.json'

def sample_values(**overrides):
    values={'previous_academic_gpa_normalized':3.5,'Age at enrollment':19,"Mother's occupation":9,"Father's occupation":9,'Scholarship holder':0,'Debtor':0,'Tuition fees up to date':1,'Curricular units 1st sem (enrolled)':6,'Curricular units 1st sem (evaluations)':8,'Curricular units 1st sem (approved)':5,'Curricular units 1st sem (grade)':12.0}
    values.update(overrides); return values

def test_exactly_eleven_fields_and_no_banned_inputs():
    assert len(FORM_FIELDS)==11
    banned=['2nd sem','Application mode','Application order','Course','Nacionality','Unemployment rate','Inflation rate','GDP',"Mother's qualification","Father's qualification",'Admission grade','Previous qualification (grade)']
    assert all(not any(term.lower() in field.lower() for term in banned) for field in FORM_FIELDS)

def test_occupation_labels_round_trip_without_raw_display_codes():
    assert OCCUPATIONS[4]=='Administrative staff'
    assert OCCUPATION_BY_LABEL['Skilled industry, construction and craft workers']==7
    assert len(OCCUPATIONS)==len(OCCUPATION_BY_LABEL)

def test_courses_passed_validation_and_input_order():
    assert validate_inputs(sample_values(**{'Curricular units 1st sem (enrolled)':4,'Curricular units 1st sem (approved)':5}))==['Courses Passed cannot be greater than Courses Enrolled.']
    assert validate_inputs(sample_values(**{'Curricular units 1st sem (enrolled)':0,'Curricular units 1st sem (approved)':0}))==[]
    assert list(make_input_frame(sample_values()).columns)==FORM_FIELDS

def test_semester_gpa_normalization_examples():
    expected={0.00:0.0,2.00:9.4375,3.00:14.15625,3.50:16.515625,4.00:18.875}
    assert {value:semester_gpa_to_source_grade(value) for value in expected}==expected

def test_saved_artifact_probability_and_locked_threshold():
    artifact=joblib.load(MODEL_PATH); metadata=json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    probability=dropout_probability(artifact,sample_values())
    assert 0<=probability<=1
    assert artifact['threshold']==metadata['threshold']==THRESHOLD==0.48

def test_frozen_model_artifact_is_unchanged():
    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()=='c228cccae5d4171afa74d9eae23e0cc88547bf9602fb6ce72626f2335b7596c1'

def test_both_result_paths():
    assert risk_label(0.479999)=='Lower Estimated Risk'
    assert risk_label(0.48)=='Elevated Estimated Risk'
    artifact=joblib.load(MODEL_PATH)
    raw=pd.read_csv(ROOT/'data'/'student_dropout.csv',sep=';'); raw.columns=raw.columns.str.strip()
    raw['previous_academic_gpa_normalized']=normalize_previous_grade(raw['Previous qualification (grade)'])
    probabilities=artifact['pipeline'].predict_proba(raw[FORM_FIELDS])[:,list(artifact['pipeline'].classes_).index(1)]
    assert risk_label(float(probabilities.min()))=='Lower Estimated Risk'
    assert risk_label(float(probabilities.max()))=='Elevated Estimated Risk'

def test_metrics_match_locked_results():
    r=json.loads(RESULTS_PATH.read_text(encoding='utf-8'))['test_metrics']
    assert round(r['precision'],4)==0.8191 and round(r['recall'],4)==0.8768
    assert round(r['f1'],4)==0.8469 and round(r['roc_auc'],4)==0.9378
    assert round(r['average_precision'],4)==0.9286 and round(r['brier_score'],4)==0.0997

def test_active_source_has_no_old_profile_ui_or_deterministic_claim():
    source=(ROOT/'app.py').read_text(encoding='utf-8').lower()
    banned=['k-means','student profile explorer','profile 1','profile 2','silhouette','davies-bouldin','clustering pca','will drop out','admission grade']
    assert all(term not in source for term in banned)
    assert 'st.image(' not in source and 'st.pyplot(' not in source

def test_three_pages_render_without_exceptions_and_form_has_11_widgets():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert list(app.radio[0].options)==['Home','Assess Dropout Risk','About Project']
    expected={'Home':'Student Dropout Early Warning System','Assess Dropout Risk':'Assess Dropout Risk','About Project':'About Project'}
    for page,title in expected.items():
        app.radio[0].set_value(page).run(); assert not app.exception; assert title in [item.value for item in app.title]
    app.radio[0].set_value('Assess Dropout Risk').run()
    assert len(app.number_input)+len(app.selectbox)==11
    gpa=next(item for item in app.number_input if item.label=='HSC / Equivalent GPA')
    assert gpa.min==2.5 and gpa.max==5.0 and gpa.step==0.01
    semester_gpa=next(item for item in app.number_input if item.label=='Semester GPA')
    assert semester_gpa.min==0.0 and semester_gpa.max==4.0 and semester_gpa.step==0.01
    semester_gpa.set_value(4.0); app.button[0].click().run()
    assert not app.exception
    assert not app.error
    app.radio[0].set_value('About Project').run()
    assert any('Anonymous Follow-Up ID' in item.value for item in app.markdown)
    assert any('first-semester academic performance' in item.value for item in app.markdown)
def guidance(**overrides):
    values=dict(semester_gpa=3.5,enrolled=6,evaluations=6,passed=6,debtor=False,
                tuition_up_to_date=True,scholarship=True,previous_gpa=4.0)
    values.update(overrides)
    return support_guidance(**values)

def test_elevated_academic_guidance():
    indicators,supports=guidance(semester_gpa=1.85,enrolled=7,evaluations=5,passed=3)
    assert indicators[0]=='Low recent academic performance — Semester GPA: 1.85 / 4.00'
    assert 'Low course completion — 3 of 7 courses passed' in indicators
    assert 'Academic advising' in supports and 'Course-specific academic support' in supports

def test_elevated_financial_guidance():
    indicators,supports=guidance(debtor=True,tuition_up_to_date=False,scholarship=False)
    assert 'Tuition fees are not currently up to date' in indicators
    assert 'Outstanding debtor status was reported' in indicators
    assert 'Financial-aid review' in supports and 'Financial counselling' in supports

def test_multiple_problems_are_prioritized_and_capped():
    indicators,supports=guidance(semester_gpa=1.2,enrolled=8,evaluations=1,passed=1,
                                 debtor=True,tuition_up_to_date=False,scholarship=False,
                                 previous_gpa=2.5)
    assert len(indicators)==3
    assert len(supports)<=4
    assert indicators[0].startswith('Low recent academic performance')

def test_lower_risk_profile_has_no_rule_warnings():
    indicators,supports=guidance()
    assert indicators==[] and supports==[]
    assert risk_label(0.2)=='Lower Estimated Risk'

def test_zero_enrolled_is_safe():
    indicators,supports=guidance(enrolled=0,evaluations=0,passed=0)
    assert 'No course enrollment was reported' in indicators
    assert supports

def test_parent_occupations_and_age_are_never_explanation_reasons():
    indicators,_=guidance(semester_gpa=1.0,enrolled=6,evaluations=1,passed=1)
    text=' '.join(indicators).lower()
    assert 'mother' not in text and 'father' not in text and 'occupation' not in text and 'age' not in text
def test_result_panels_show_guidance_only_for_elevated_risk():
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    app.radio[0].set_value('Assess Dropout Risk').run()
    numbers={item.label:item for item in app.number_input}
    selections={item.label:item for item in app.selectbox}
    numbers['HSC / Equivalent GPA'].set_value(2.5)
    numbers['Age at Enrollment'].set_value(70)
    numbers['Courses Enrolled'].set_value(7)
    numbers['Evaluations Completed'].set_value(1)
    numbers['Courses Passed'].set_value(0)
    numbers['Semester GPA'].set_value(0.0)
    selections['Scholarship Holder'].set_value('No')
    selections['Debtor'].set_value('Yes')
    selections['Tuition Fees Up to Date'].set_value('No')
    app.button[0].click().run()
    rendered=' '.join(item.value for item in app.markdown)
    assert 'Elevated Estimated Risk' in rendered
    assert 'Key Risk Indicators' in rendered
    assert 'Possible Institute Support' in rendered
    assert 'confirmed causes of dropout' in ' '.join(item.value for item in app.caption)