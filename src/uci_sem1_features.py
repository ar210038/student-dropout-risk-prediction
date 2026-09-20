"""Leakage-safe derived features for the UCI Semester-1 model."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

FINAL_FEATURES=['Age at enrollment','Previous qualification (grade)','Admission grade',"Mother's occupation","Father's occupation",'Scholarship holder','Debtor','Tuition fees up to date','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)','Curricular units 1st sem (grade)','Semester pass rate']

class SemesterFeatureBuilder(BaseEstimator, TransformerMixin):
    """Calculate approved/enrolled safely; a zero-course semester maps to rate 0."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        frame=X.copy()
        enrolled=frame['Curricular units 1st sem (enrolled)'].astype(float)
        approved=frame.pop('Curricular units 1st sem (approved)').astype(float)
        frame['Semester pass rate']=np.divide(approved,enrolled,out=np.zeros(len(frame),dtype=float),where=enrolled.to_numpy()!=0)
        return frame[FINAL_FEATURES]
