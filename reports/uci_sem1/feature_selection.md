# Feature Selection

## Full leakage-safe pool

The 18-field candidate pool was: Age at enrollment; Previous qualification grade; Admission grade; Gender; Daytime/evening attendance; Mother's and Father's occupation; Mother's and Father's qualification; Scholarship holder; Debtor; Tuition fees up to date; first-semester enrolled, evaluations, approved, grade, credited, and without-evaluation counts.

Selection was performed using training data only. Repeated-CV stability, permutation importance, form clarity, redundancy, and timing were considered. No holdout metric was used to select fields.

## Full versus simplified

| Logistic Regression form | Inputs | ROC-AUC | PR-AUC | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Full candidate | 18 | 0.9248 | 0.9208 | 0.8307 | 0.8390 |
| Simplified | 12 | 0.9114 | 0.9031 | 0.8109 | 0.8261 |
| Difference, simplified − full | −6 | −0.0133 | −0.0177 | −0.0198 | −0.0130 |

The simplified form retains most discrimination while removing six lower-value or less practical inputs. This is a deliberate usability tradeoff, not a claim that omitted factors are irrelevant.

## Derived Semester pass rate

`Semester pass rate = first-semester approved units / first-semester enrolled units`. When enrolled units equal zero, the rate is defined as 0 to avoid division by zero. The website will ask for the two understandable counts and calculate the rate inside the saved pipeline.

## Final model fields

The model representation contains 11 direct fields plus Semester pass rate. The actual proposed form has 12 entries because Approved units is entered to calculate the derived rate.

Training-set permutation importance ranked the associated raw form fields approximately as follows: Approved units, Enrolled units, Tuition fees up to date, Semester grade, Scholarship holder, Evaluations completed, Age, Father's occupation, Mother's occupation, Admission grade, Previous grade, and Debtor. These fields are associated with dropout prediction; they do not cause dropout.
