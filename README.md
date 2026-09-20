# Student Dropout Early Warning System Using Machine Learning

This academic project estimates elevated risk of later university dropout after a student completes Semester 1. It supports timely review—not automatic decisions or certainty about a student's future.

## Prediction point and target

The prediction point is the **end of the first semester**. The model uses background, financial, and Semester-1 performance information available then.

- Positive class: Dropout = 1
- Negative class: Graduate = 0
- Enrolled students are excluded because their final outcomes are unresolved
- All Semester-2 and final-outcome information is excluded

## Dataset

The project uses the UCI Machine Learning Repository dataset **Predict Students' Dropout and Academic Success**.

| Cohort | Students |
|---|---:|
| Original dataset | 4,424 |
| Graduate | 2,209 |
| Dropout | 1,421 |
| Enrolled excluded | 794 |
| Resolved modeling cohort | 3,630 |

Dropout prevalence is 39.15%.

## Final 12-input form

1. Age at Enrollment
2. Previous Academic Grade
3. Admission Grade
4. Mother's Occupation
5. Father's Occupation
6. Scholarship Holder
7. Debtor
8. Tuition Fees Up to Date
9. Courses Enrolled
10. Evaluations Completed
11. Courses Passed
12. Average Semester Grade

Occupation codes are presented as documented readable labels. Semester pass rate is calculated inside the pipeline as `Courses Passed / Courses Enrolled`; zero enrolled courses produces a rate of zero.

## Model development and results

Preprocessing and feature engineering are contained in a scikit-learn pipeline. Logistic Regression, Random Forest, and XGBoost were compared with repeated stratified cross-validation on training data. The holdout was not used for feature or threshold selection.

The selected model is **Random Forest**, with a locked threshold of **0.48**.

| Held-out metric | Result |
|---|---:|
| Accuracy | 88.3% |
| Balanced accuracy | 88.1% |
| Precision | 83.5% |
| Recall | 87.3% |
| F1 | 85.4% |
| ROC-AUC | 93.9% |
| PR-AUC | 93.0% |
| Brier score | 0.0967 |

Confusion matrix: 393 correctly identified graduates, 49 false early warnings, 36 missed dropout cases, and 248 correctly identified dropout cases.

## Streamlit interface

The four pages are Home, Assess Dropout Risk, Model Performance, and About & Methodology. The assessment page loads the frozen artifact from `models/uci_sem1/`, calls `predict_proba()`, reads the Dropout=1 probability, and applies the saved threshold. It never retrains.

## Leakage controls

The final model excludes Target, every Semester-2 variable, final-outcome information, Application Mode, Application Order, Course code, Nationality, unemployment, inflation, and GDP. No information occurring after the prediction point is used.

## Local use

Python 3.12 is the deployment target. The artifact was trained under Python 3.14.5 with scikit-learn 1.9.1; Python 3.12 must be tested independently before deployment.

```bash
python -m pip install -r requirements.txt
python -m pytest -q --basetemp=.pytest_tmp
python -m streamlit run app.py
```

## Limitations

- Data comes from Portuguese higher education and has not been validated in Bangladesh.
- Academic systems may differ between countries.
- Occupation categories originate from the source dataset.
- Probability calibration may require local recalibration.
- Predictions are estimates, not certainties.
- Outputs should support, not replace, human judgement.
- Operational deployment requires local longitudinal validation or retraining.
