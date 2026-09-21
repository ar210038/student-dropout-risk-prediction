"""Train the isolated UCI end-of-first-semester early-warning candidate."""
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from src.uci_sem1_features import FINAL_FEATURES, SemesterFeatureBuilder

ROOT=Path(__file__).resolve().parents[1]
DATA_PATH=ROOT/'data'/'student_dropout.csv'; OUT=ROOT/'reports'/'uci_sem1'; FIG=OUT/'figures'; MODEL_DIR=ROOT/'models'/'uci_sem1'
MODEL_PATH=MODEL_DIR/'dropout_early_warning_model.joblib'; META_PATH=MODEL_DIR/'model_metadata.json'; RESULTS_PATH=OUT/'training_results.json'
TARGET='Target'; RANDOM_STATE=42
BANNED={'Application mode','Application order','Course','Nacionality','Unemployment rate','Inflation rate','GDP'}
FULL_FEATURES=['Age at enrollment','Previous qualification (grade)','Admission grade','Gender','Daytime/evening attendance','Mother\'s occupation','Father\'s occupation','Mother\'s qualification','Father\'s qualification','Scholarship holder','Debtor','Tuition fees up to date','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (approved)','Curricular units 1st sem (grade)','Curricular units 1st sem (credited)','Curricular units 1st sem (without evaluations)']
RAW_FOR_FINAL=[f for f in FINAL_FEATURES if f!='Semester pass rate']+['Curricular units 1st sem (approved)']
CATEGORICAL_FULL=['Gender','Daytime/evening attendance','Mother\'s occupation','Father\'s occupation','Mother\'s qualification','Father\'s qualification','Scholarship holder','Debtor','Tuition fees up to date']
CATEGORICAL_FINAL=['Mother\'s occupation','Father\'s occupation','Scholarship holder','Debtor','Tuition fees up to date']
SCORING={'roc_auc':'roc_auc','average_precision':'average_precision','precision':'precision','recall':'recall','f1':'f1','balanced_accuracy':'balanced_accuracy'}

def load_data():
    d=pd.read_csv(DATA_PATH,sep=';'); d.columns=d.columns.str.strip(); return d

def prepare(d):
    cohort=d[d[TARGET].isin(['Graduate','Dropout'])].copy(); return cohort,cohort[TARGET].map({'Graduate':0,'Dropout':1}).astype('int8')

def preprocess(features,categorical):
    numeric=[f for f in features if f not in categorical]
    return ColumnTransformer([('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical),('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric)])

def pipeline(name,final=True):
    features=FINAL_FEATURES if final else FULL_FEATURES; categorical=CATEGORICAL_FINAL if final else CATEGORICAL_FULL
    if name=='Logistic Regression': model=LogisticRegression(max_iter=3000,class_weight='balanced',C=0.5,random_state=42)
    elif name=='Random Forest': model=RandomForestClassifier(n_estimators=500,max_depth=12,min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=-1,random_state=42)
    else: model=XGBClassifier(n_estimators=350,max_depth=3,learning_rate=.035,subsample=.85,colsample_bytree=.85,min_child_weight=5,reg_lambda=2,eval_metric='logloss',n_jobs=-1,random_state=42,scale_pos_weight=2209/1421)
    steps=[]
    if final: steps.append(('derive',SemesterFeatureBuilder()))
    steps.extend([('preprocess',preprocess(features,categorical)),('model',model)])
    return Pipeline(steps)

def cv_summary(model,X,y,cv):
    s=cross_validate(model,X,y,cv=cv,scoring=SCORING,n_jobs=1)
    return {m:{'mean':float(np.mean(s['test_'+m])),'std':float(np.std(s['test_'+m],ddof=1))} for m in SCORING}

def metrics(y,p,t):
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
    return {'threshold':float(t),'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'precision':precision_score(y,pred),'recall':recall_score(y,pred),'f1':f1_score(y,pred),'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),'brier_score':brier_score_loss(y,p),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'predicted_positive_rate':float(pred.mean())}

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); MODEL_DIR.mkdir(parents=True,exist_ok=True)
    raw=load_data(); cohort,y=prepare(raw); X=cohort.drop(columns=[TARGET])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    repeated=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=42)
    full_lr=Pipeline([('preprocess',preprocess(FULL_FEATURES,CATEGORICAL_FULL)),('model',LogisticRegression(max_iter=3000,class_weight='balanced',C=.5,random_state=42))])
    feature_comparison={'full_18_logistic':cv_summary(full_lr,Xtr[FULL_FEATURES],ytr,repeated),'simplified_12_logistic':cv_summary(pipeline('Logistic Regression'),Xtr[RAW_FOR_FINAL],ytr,repeated)}
    models={name:pipeline(name) for name in ['Logistic Regression','Random Forest','XGBoost']}
    cv_results={name:cv_summary(model,Xtr[RAW_FOR_FINAL],ytr,repeated) for name,model in models.items()}
    selected=max(cv_results,key=lambda n:(cv_results[n]['roc_auc']['mean'],cv_results[n]['average_precision']['mean'],cv_results[n]['f1']['mean']))
    selected_model=models[selected]; folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof=cross_val_predict(selected_model,Xtr[RAW_FOR_FINAL],ytr,cv=folds,method='predict_proba',n_jobs=1)[:,1]
    thresholds=[]
    for t in np.arange(.20,.71,.02): thresholds.append(metrics(ytr,oof,float(round(t,2))))
    eligible=[r for r in thresholds if r['recall']>=.78 and r['precision']>=.60]
    chosen=max(eligible or thresholds,key=lambda r:(r['f1'],r['balanced_accuracy']))
    selected_model.fit(Xtr[RAW_FOR_FINAL],ytr); test_p=selected_model.predict_proba(Xte[RAW_FOR_FINAL])[:,1]; test=metrics(yte,test_p,chosen['threshold'])
    perm=permutation_importance(selected_model,Xtr[RAW_FOR_FINAL],ytr,n_repeats=10,random_state=42,scoring='roc_auc',n_jobs=1)
    importance=sorted([{'feature':f,'importance_mean':float(m),'importance_std':float(s)} for f,m,s in zip(RAW_FOR_FINAL,perm.importances_mean,perm.importances_std)],key=lambda z:z['importance_mean'],reverse=True)
    from sklearn.calibration import calibration_curve
    frac,mean=calibration_curve(yte,test_p,n_bins=10,strategy='quantile'); fig,ax=plt.subplots(); ax.plot(mean,frac,'o-',label=selected); ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='Mean predicted probability',ylabel='Observed dropout fraction',title='Holdout calibration curve'); ax.legend(); fig.tight_layout(); fig.savefig(FIG/'calibration_curve.png',dpi=180); plt.close(fig)
    cm=np.array([[test['tn'],test['fp']],[test['fn'],test['tp']]]); fig,ax=plt.subplots(); im=ax.imshow(cm,cmap='Blues'); [ax.text(j,i,str(cm[i,j]),ha='center',va='center') for i in range(2) for j in range(2)]; ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Graduate','Dropout'],yticklabels=['Graduate','Dropout'],xlabel='Predicted',ylabel='Actual',title='Semester-1 holdout confusion matrix'); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(FIG/'confusion_matrix.png',dpi=180); plt.close(fig)
    artifact={'pipeline':selected_model,'threshold':chosen['threshold'],'features':FINAL_FEATURES,'raw_input_features':RAW_FOR_FINAL}; joblib.dump(artifact,MODEL_PATH)
    results={'dataset':{'rows':len(raw),'predictors':len(raw.columns)-1,'target_counts':raw[TARGET].value_counts().to_dict(),'modeling_cohort':len(cohort),'dropout_prevalence':float(y.mean()),'train_size':len(ytr),'test_size':len(yte)},'prediction_point':'End of first semester','full_features':FULL_FEATURES,'final_features':FINAL_FEATURES,'raw_input_features':RAW_FOR_FINAL,'excluded':sorted(BANNED|{TARGET}|{c for c in raw.columns if c.startswith('Curricular units 2nd sem')}),'feature_comparison':feature_comparison,'model_cv':cv_results,'selected_model':selected,'threshold_candidates':thresholds,'selected_threshold_training_metrics':chosen,'test_metrics':test,'permutation_importance':importance,'calibration_points':{'mean_predicted':mean.tolist(),'observed_fraction':frac.tolist()},'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'platform':platform.platform()}}
    RESULTS_PATH.write_text(json.dumps(results,indent=2),encoding='utf-8')
    metadata={'title':'Student Dropout Early Warning System Using Machine Learning','prediction_point':'End of first semester','model':selected,'threshold':chosen['threshold'],'features':FINAL_FEATURES,'raw_input_features':RAW_FOR_FINAL,'target_mapping':{'Graduate':0,'Dropout':1},'excluded_outcome':'Enrolled','versions':results['versions'],'limitation':'Trained on Portuguese higher-education data; requires external validation or local retraining before operational use elsewhere.'}; META_PATH.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
