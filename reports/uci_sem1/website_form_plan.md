# Future Semester-1 Website Form Plan

## Workflow

Student completes Semester 1 → department enters available information → saved model estimates later dropout risk → result is **Lower Estimated Risk** or **Elevated Estimated Risk** → staff may consider supportive follow-up.

The future application must load `models/uci_sem1/dropout_early_warning_model.joblib`; it must not retrain. Migration is not part of this round.

## Exact 12 fields

### Student background

1. Age at Enrollment
2. Previous Academic Grade
3. Admission Grade

### Family background

4. Mother's Occupation
5. Father's Occupation

### Financial status

6. Scholarship Holder
7. Debtor
8. Tuition Fees Up to Date

### First-semester performance

9. Courses Enrolled
10. Evaluations Completed
11. Courses Passed
12. Average Semester Grade

Courses Passed and Courses Enrolled are converted internally to Semester pass rate. Parent occupation codes must be rendered with the official UCI occupation descriptions already preserved in the archived UCI interface; raw numeric codes must never be shown.

## Result wording

The locked threshold is 0.48. A future interface may display the estimated percentage because the Brier score is 0.0967, but must say:

> This is an early-warning estimate based on patterns in the training dataset. It does not determine whether a student will actually drop out.

It must also display:

> The model was trained on a publicly available Portuguese higher-education dataset. It is an academic prototype. Before operational use in Bangladesh or another country, it should be externally validated or retrained using local longitudinal university data.

The output must never say that a student “will drop out,” and it must not trigger punitive or automatic decisions.
