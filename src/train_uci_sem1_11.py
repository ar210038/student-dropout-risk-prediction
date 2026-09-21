"""Controlled migration from the validated 12-input model to the 11-input form."""
from __future__ import annotations
import json,platform
from pathlib import Path
import joblib,matplotlib.pyplot as plt,numpy as np,pandas as pd,sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,brier_score_loss,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold,cross_val_predict,cross_validate,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.uci_sem1_features import FINAL_FEATURES,NORMALIZED_GPA_FEATURE,RAW_INPUT_FEATURES,SemesterFeatureBuilder,normalize_previous_grade

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'student_dropout.csv'; REPORT=ROOT/'reports'/'uci_sem1'; FIG=REPORT/'figures'; MODEL_DIR=ROOT/'models'/'uci_sem1'
MODEL_PATH=MODEL_DIR/'dropout_early_warning_model.joblib'; META_PATH=MODEL_DIR/'model_metadata.json'; RESULTS=REPORT/'training_results_11_input.json'; BASELINE=REPORT/'training_results.json'
CATEGORICAL=["Mother's occupation","Father's occupation",'Scholarship holder','Debtor','Tuition fees up to date']; SCORING={'roc_auc':'roc_auc','average_precision':'average_precision','precision':'precision','recall':'recall','f1':'f1','balanced_accuracy':'balanced_accuracy'}

def load_frame():
    d=pd.read_csv(DATA,sep=';'); d.columns=d.columns.str.strip(); d=d[d.Target.isin(['Graduate','Dropout'])].copy(); d[NORMALIZED_GPA_FEATURE]=normalize_previous_grade(d['Previous qualification (grade)']); return d,d.Target.map({'Graduate':0,'Dropout':1}).astype('int8')
def make_pipeline():
    numeric=[f for f in FINAL_FEATURES if f not in CATEGORICAL]
    prep=ColumnTransformer([('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAL),('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric)])
    forest=RandomForestClassifier(n_estimators=500,max_depth=12,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=42)
    return Pipeline([('derive',SemesterFeatureBuilder()),('preprocess',prep),('model',forest)])
def cv_summary(model,X,y,cv):
    s=cross_validate(model,X,y,cv=cv,scoring=SCORING,n_jobs=1); return {m:{'mean':float(np.mean(s['test_'+m])),'std':float(np.std(s['test_'+m],ddof=1))} for m in SCORING}
def metrics(y,p,t):
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred).ravel(); return {'threshold':float(t),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'precision':precision_score(y,pred),'recall':recall_score(y,pred),'f1':f1_score(y,pred),'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),'brier_score':brier_score_loss(y,p),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'predicted_positive_rate':float(pred.mean())}
def main():
    frame,y=load_frame(); Xtr,Xte,ytr,yte=train_test_split(frame,y,test_size=.2,stratify=y,random_state=42); model=make_pipeline()
    cv=cv_summary(model,Xtr[RAW_INPUT_FEATURES],ytr,RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=42)); baseline=json.loads(BASELINE.read_text(encoding='utf-8'))['model_cv']['Random Forest']
    gate=(cv['roc_auc']['mean']>=.92 and cv['roc_auc']['mean']>=baseline['roc_auc']['mean']-.01 and cv['f1']['mean']>=baseline['f1']['mean']-.01 and cv['recall']['mean']>=baseline['recall']['mean']-.01)
    if not gate: raise RuntimeError(f'11-input CV gate failed: {cv}')
    folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); oof=cross_val_predict(model,Xtr[RAW_INPUT_FEATURES],ytr,cv=folds,method='predict_proba',n_jobs=1)[:,1]
    candidates=[metrics(ytr,oof,float(round(t,2))) for t in np.arange(.20,.71,.02)]; eligible=[r for r in candidates if r['recall']>=.78 and r['precision']>=.60]; chosen=max(eligible or candidates,key=lambda r:(r['f1'],r['balanced_accuracy']))
    model.fit(Xtr[RAW_INPUT_FEATURES],ytr); test_p=model.predict_proba(Xte[RAW_INPUT_FEATURES])[:,1]; test=metrics(yte,test_p,chosen['threshold'])
    perm=permutation_importance(model,Xtr[RAW_INPUT_FEATURES],ytr,n_repeats=10,random_state=42,scoring='roc_auc',n_jobs=1); importance=sorted([{'feature':f,'importance_mean':float(m),'importance_std':float(s)} for f,m,s in zip(RAW_INPUT_FEATURES,perm.importances_mean,perm.importances_std)],key=lambda r:r['importance_mean'],reverse=True)
    frac,mean=calibration_curve(yte,test_p,n_bins=10,strategy='quantile'); fig,ax=plt.subplots(); ax.plot(mean,frac,'o-',label='Random Forest'); ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='Mean predicted probability',ylabel='Observed dropout fraction',title='Holdout calibration curve — final 11-input model'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'calibration_curve_11_input.png',dpi=180); plt.close(fig)
    cm=np.array([[test['tn'],test['fp']],[test['fn'],test['tp']]]); fig,ax=plt.subplots(); im=ax.imshow(cm,cmap='Blues'); [ax.text(j,i,str(cm[i,j]),ha='center',va='center') for i in range(2) for j in range(2)]; ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Graduate','Dropout'],yticklabels=['Graduate','Dropout'],xlabel='Predicted',ylabel='Actual',title='Final 11-input holdout confusion matrix'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(FIG/'confusion_matrix_11_input.png',dpi=180); plt.close(fig)
    comparison={m:cv[m]['mean']-baseline[m]['mean'] for m in SCORING}
    result={'decision_gate_passed':gate,'normalized_gpa_range':{'min':float(frame[NORMALIZED_GPA_FEATURE].min()),'max':float(frame[NORMALIZED_GPA_FEATURE].max()),'formula':'Previous qualification (grade) / 190 * 5'},'features':FINAL_FEATURES,'raw_input_features':RAW_INPUT_FEATURES,'cv':cv,'baseline_12_input_cv':baseline,'cv_difference_11_minus_12':comparison,'threshold_candidates':candidates,'selected_threshold_training_metrics':chosen,'test_metrics':test,'permutation_importance':importance,'holdout_note':'Previously consulted during project iteration; final project benchmark, not pristine external validation.','versions':{'python':platform.python_version(),'scikit_learn':sklearn.__version__}}
    RESULTS.write_text(json.dumps(result,indent=2),encoding='utf-8'); joblib.dump({'pipeline':model,'threshold':chosen['threshold'],'features':FINAL_FEATURES,'raw_input_features':RAW_INPUT_FEATURES},MODEL_PATH)
    META_PATH.write_text(json.dumps({'title':'Student Dropout Early Warning System Using Machine Learning','model':'Random Forest','prediction_point':'End of first semester','threshold':chosen['threshold'],'features':FINAL_FEATURES,'raw_input_features':RAW_INPUT_FEATURES,'normalized_gpa_range':result['normalized_gpa_range'],'target_mapping':{'Graduate':0,'Dropout':1},'excluded_outcome':'Enrolled','versions':result['versions'],'limitation':'Scale normalization is numerical only; cross-country academic equivalence is not claimed.'},indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
