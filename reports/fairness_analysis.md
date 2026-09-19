# Fairness and Sensitivity Analysis

This is a diagnostic review of the final held-out test predictions from **Logistic Regression** at threshold **0.40**. It does not establish that the model is fair. The test set was examined only after model and threshold selection and was not used for tuning.

Groups are reported only when they contain at least 30 students, 10 dropouts, and 10 graduates.

| Attribute | Group | Sample count | Dropout prevalence | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Gender | Female | 438 | 0.297 | 0.580 | 0.754 | 0.656 |
| Gender | Male | 288 | 0.535 | 0.653 | 0.903 | 0.757 |
| Scholarship holder | No | 552 | 0.467 | 0.621 | 0.864 | 0.723 |
| Scholarship holder | Yes | 174 | 0.149 | 0.609 | 0.538 | 0.571 |
| Debtor | No | 642 | 0.346 | 0.572 | 0.802 | 0.668 |
| Debtor | Yes | 84 | 0.738 | 0.831 | 0.952 | 0.887 |
| Age group | 17-20 | 422 | 0.268 | 0.519 | 0.743 | 0.611 |
| Age group | 21-25 | 130 | 0.423 | 0.600 | 0.818 | 0.692 |
| Age group | 26+ | 174 | 0.667 | 0.745 | 0.931 | 0.828 |

## Notable differences

- Age group: recall gap 0.188, precision gap 0.226.
- Debtor: recall gap 0.150, precision gap 0.259.
- Scholarship holder: recall gap 0.326, precision gap 0.012.

Differences may reflect prevalence, sample composition, historical processes, or random test-split variation. They should not be interpreted as inherent differences in student ability.

## Groups not reported because of sample size

- None

## Use limitations

- Gender, age, debt, scholarship, nationality, disability-related information, and family background are sensitive or proxy variables.
- Predictions should be used to offer support, never to deny admission, aid, or educational opportunity.
- Subgroup metrics need external validation on the intended institution and future cohorts.
- Small-group metrics should remain suppressed or explicitly qualified.
