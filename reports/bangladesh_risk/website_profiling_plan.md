# Website Profiling Plan

## Decision

**MIGRATE TO PROFILING WEBSITE**, but only as a descriptive, educational profile explorer. Do not retain the current supervised-prediction framing, do not show an individual dropout probability, and do not describe either cluster as a risk tier.

## Proposed user flow

1. Explain that the tool compares a student's answers with two broad patterns found in a 351-response Bangladesh survey.
2. Collect the 12 exact model inputs: age group, academic level, personal income, employment type, internet quality, living arrangement, study routine, family income, scholarship, study-space quality, academic-resource access, and current GPA.
3. Assign the fitted descriptive profile deterministically.
4. Show the profile name, the user's matching answers, dominant sample characteristics, and profile size.
5. Show the observed profile-level elevated-intention percentage only in a clearly labelled “sample context” section with numerator/denominator and the statement that cluster membership was not meaningfully associated with elevated intention.
6. Offer general, non-diagnostic student-support guidance and a link to institutional services; do not prescribe action from cluster membership.

## Required wording and boundaries

- Title: “Bangladesh University Student Profile Explorer”.
- Primary output: “Your answers most closely match …”, not “Your dropout risk is …”.
- Required disclosure: “This profile is a descriptive similarity grouping, not a prediction, diagnosis, or probability of dropping out.”
- Required evidence note: “In this sample, profile membership had negligible association with elevated dropout intention (Cramér's V=0.038; p=0.557).”
- Never label profiles low/high risk or rank them.
- Never expose a cluster ID without its descriptive label.
- Never use the target, a target-derived field, or academic-overwhelm response as an input.
- Do not collect names, student IDs, email addresses, or timestamps.

## Technical integration

Load `models/bangladesh_student_profiler.joblib` and validate inputs against `models/bangladesh_student_profiler_metadata.json`. The saved pipeline performs encoding, scaling, and K-Means assignment. Use `reports/bangladesh_risk/profile_definitions.json` for display text and aggregate statistics. Validate schema and allowed category values before prediction; log no personal answers by default.

Before migration, replace the present app in a separate reviewed change, add UI-level exclusion and privacy tests, and conduct a language/accessibility review. The current round deliberately makes no Streamlit changes and performs no deployment.
