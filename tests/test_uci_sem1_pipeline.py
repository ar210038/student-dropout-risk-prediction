import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.train_uci_sem1 import (BANNED, DATA_PATH, FINAL_FEATURES, FULL_FEATURES,
    META_PATH, MODEL_PATH, RAW_FOR_FINAL, RESULTS_PATH, TARGET, load_data, prepare)
from src.uci_sem1_features import SemesterFeatureBuilder

def test_dataset_and_target_cohort():
    raw=load_data(); cohort,y=prepare(raw)
    assert raw.shape==(4424,37)
    assert raw[TARGET].value_counts().to_dict()=={'Graduate':2209,'Dropout':1421,'Enrolled':794}
    assert len(cohort)==3630 and int(y.sum())==1421
    assert set(y.unique())=={0,1}

def test_feature_policy_has_no_leakage_or_banned_fields():
    assert TARGET not in FULL_FEATURES+FINAL_FEATURES+RAW_FOR_FINAL
    assert all(not f.startswith('Curricular units 2nd sem') for f in FULL_FEATURES+FINAL_FEATURES+RAW_FOR_FINAL)
    assert BANNED.isdisjoint(FULL_FEATURES)
    assert len(RAW_FOR_FINAL)==12

def test_derived_pass_rate_and_zero_denominator():
    frame=pd.DataFrame({'Curricular units 1st sem (enrolled)':[4,0], 'Curricular units 1st sem (approved)':[3,0], **{f:[1,1] for f in RAW_FOR_FINAL if f not in {'Curricular units 1st sem (enrolled)','Curricular units 1st sem (approved)'}}})
    result=SemesterFeatureBuilder().transform(frame)
    assert result['Semester pass rate'].tolist()==[.75,0.0]
    assert np.isfinite(result['Semester pass rate']).all()

def test_saved_artifact_round_trip_probability_and_threshold():
    artifact=joblib.load(MODEL_PATH); metadata=json.loads(META_PATH.read_text(encoding='utf-8'))
    raw=load_data(); cohort,_=prepare(raw); row=cohort[RAW_FOR_FINAL].iloc[:2]
    probabilities=artifact['pipeline'].predict_proba(row)[:,1]
    assert isinstance(artifact['pipeline'],Pipeline)
    assert np.all((probabilities>=0)&(probabilities<=1))
    assert artifact['threshold']==metadata['threshold']==0.48
    assert artifact['features']==FINAL_FEATURES

def test_saved_results_are_locked_and_holdout_is_single_use():
    results=json.loads(RESULTS_PATH.read_text(encoding='utf-8'))
    assert results['dataset']['train_size']==2904 and results['dataset']['test_size']==726
    assert results['selected_model']=='Random Forest'
    assert results['test_metrics']['tn']+results['test_metrics']['fp']+results['test_metrics']['fn']+results['test_metrics']['tp']==726
