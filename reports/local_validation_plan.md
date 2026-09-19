# Local Validation Plan

## Objective

Determine whether the existing Portuguese-trained dropout-risk model generalizes to students at a Bangladeshi university. This is an external validation study, not a model-training exercise. The model, preprocessing pipeline, feature order, and 0.40 decision threshold must remain frozen throughout the primary validation.

## Required Local Data

Use anonymized historical student records with resolved outcomes: students who ultimately graduated or dropped out. Capture only information that was genuinely available at enrollment and that can be ethically and legally used for research.

The preferred starting schema is the model's 24 enrollment-time features. A local institution should not force every field into the study when the concept or categories do not translate meaningfully. For each available field, record:

- the local source system and data owner;
- the date at which the value becomes available;
- its local definition and allowed values;
- missingness and data-quality rules;
- the documented mapping, if any, to the UCI feature;
- the legal and ethical basis for its research use.

The outcome definition also needs a fixed observation window and an institutionally agreed definition of dropout. Transfers, temporary leave, readmission, and students still enrolled at the cutoff should be handled explicitly. Students with unresolved outcomes should not be labelled as graduates or dropouts.

## Suggested Sample

Aim for several hundred students with resolved outcomes if feasible. More cases are preferable, especially enough dropout cases to estimate recall with useful uncertainty. No fixed sample size is statistically guaranteed to be sufficient: it depends on the local dropout rate, subgroup analyses, missingness, and desired confidence-interval width.

Before extracting records, a statistician or research-methods supervisor should estimate the required sample from the anticipated number of dropout outcomes and the precision needed for recall and calibration estimates. Report confidence intervals, not only point estimates. If the available dropout count is small, treat the work as a pilot and avoid strong generalization claims.

## Feature Mapping

Create a reviewed mapping dictionary before running the model. Likely compatibility problems include:

- **Course codes:** Portuguese degree codes do not directly represent Bangladeshi programmes. Map only when academic meaning is defensible; otherwise mark the feature as incompatible.
- **Application modes:** Portuguese admission routes and ordinances are institution-specific. Do not assign a superficially similar Portuguese code without written justification.
- **Qualification categories:** School systems, degree names, and grading scales differ. Document any equivalence and grade conversion.
- **Nationality:** A Portuguese-centered coding scheme is inappropriate as a default local representation. Review whether the feature should be mapped, treated as unavailable, or excluded from a separate research comparison.
- **Parental qualifications and occupations:** Local education and occupation classifications may not align with the source taxonomy. Use a crosswalk reviewed by subject-matter experts.
- **Economic indicators:** Confirm the geographic level, reference year, units, and source for unemployment, inflation, and GDP values. Do not substitute contemporary national values for historical enrollment-period values.

Never assign Portuguese category codes merely to make the pipeline run. Maintain a mapping log with the original local value, mapped value, rationale, reviewer, version, and unresolved cases. Quantify how many local records are unmappable or outside the original training range.

If essential fields cannot be mapped defensibly, do not report a standard external-validation result as if the schemas matched. Instead, document the incompatibility and treat adapting or rebuilding a local model as separate future work.

## Validation Method

1. Approve a protocol, outcome definition, cohort dates, feature crosswalk, and analysis plan before viewing performance results.
2. Extract and anonymize the resolved local cohort. Remove names, student IDs, contact details, and free-text identifiers.
3. Apply documented transformations outside the model solely to create the exact 24-column input schema. Keep raw and mapped data separate and auditable.
4. Load the existing `models/dropout_model.joblib` pipeline and `models/model_metadata.json` metadata.
5. Keep the fitted pipeline and threshold frozen. Do not refit encoders, scalers, coefficients, or the threshold on the validation cohort.
6. Generate dropout probabilities with `predict_proba` and binary decisions at 0.40.
7. Report ROC-AUC, precision, recall, F1-score, and the confusion matrix with confidence intervals where practical.
8. Assess calibration with a calibration plot and a proper scoring measure such as Brier score when sample size permits. Compare predicted probability ranges with observed dropout rates.
9. Compare local results with the original held-out results while emphasizing differences in population, definitions, and data collection.
10. Record mapping coverage, missingness, out-of-range values, and any excluded records alongside performance metrics.

Do not select a new threshold after examining the same validation outcomes and then present it as unbiased validation. Any exploratory local threshold analysis must be labelled separately and confirmed on a future independent cohort.

If enough high-quality local data later becomes available, a locally trained model may be developed and compared with the frozen external model. That would be a separate future study with its own training, tuning, and held-out evaluation design.

## Fairness Review

Review performance across relevant groups only when there are enough students and dropout outcomes for estimates to be meaningful. Potential groups may include gender, scholarship status, disability or special-needs status, programme, and socioeconomic indicators, subject to ethics approval and local law.

For each adequately represented group, consider recall, false-positive rate, precision, calibration, sample size, and uncertainty. Suppress or combine very small groups to reduce re-identification risk and unstable estimates. Similar-looking point estimates do not prove fairness, and different metrics can reveal different harms. Include qualitative review by student-support staff and, where possible, student representatives.

Sensitive attributes should be evaluated for fairness and necessity. Their presence in the source model does not automatically justify operational use locally.

## Privacy and Governance

- Use anonymized or properly de-identified student records.
- Do not include names, student IDs, phone numbers, email addresses, national identifiers, precise addresses, or identifying free text.
- Store the re-identification key separately, if one is required at all, with access limited to an authorized data custodian.
- Apply institutional ethics approval, data-protection rules, retention limits, access controls, and secure storage.
- Report aggregate results and prevent publication of small cells that could identify individuals.
- Keep an audit trail of dataset versions, mapping decisions, model version, metadata, and analysis code.
- Use predictions only for support-oriented research. Do not use them as the sole basis for admission, discipline, financial, or academic decisions.

## Decision Criteria and Reporting

Before validation begins, define what results would justify further research rather than immediate deployment. The report should distinguish discrimination, classification at the frozen threshold, calibration, fairness, mapping coverage, and operational feasibility.

Any decision to pilot the model should require acceptable local evidence, human oversight, a clear support pathway, student communication, monitoring for harm, and a process for appeal or correction. If mapping quality, calibration, recall, or subgroup reliability is inadequate, the appropriate conclusion is that the imported model is not validated for local use.
