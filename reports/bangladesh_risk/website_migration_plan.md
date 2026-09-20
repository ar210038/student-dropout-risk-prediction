# Proposed Bangladesh Risk-Assessment Website Form

This is a plan only. The current Streamlit application has not been changed. Migration is not recommended with the present model performance.

| Website Label | Input Type | Allowed Values | Model Feature |
|---|---|---|---|
| Academic Level | Select | Undergraduate Year 1; Undergraduate Years 2–4; Master's | `academic_level` |
| Sleep Duration | Select | Under 5 hours; 5–6 hours; 7–8 hours; Over 8 hours | `sleep_duration` |
| Employment / Tuition Work | Select | No; Tuition/Coaching; Part-time/Full-time Job; Freelancing | `employment_type` |
| Commute Time | Select | Under 30 minutes; 30–60 minutes; Over 1 hour | `commute_time` |
| Internet Quality | Rating | 1 Very Poor through 5 Excellent | `internet_quality` |
| Study Space | Select | No dedicated space; Very noisy; Noisy; Moderate noise; Quiet; Very quiet | `study_space_quality` |
| Current GPA | Select | Under 2.5; 2.5–3.0; 3.1–3.5; Above 3.5 | `current_gpa` |
| Study Routine | Select | Regularly; Sporadically; Only before exams | `study_routine` |
| Attendance | Select | Under 50%; 50–75%; Above 75% | `attendance` |
| Scholarship / Stipend | Select | None; Government; University merit-based; Private/donor-funded | `scholarship` |
| Academic Resource Access | Select | No access; Rarely; Weekly; Daily | `academic_resource_access` |
| Study Material Satisfaction | Rating | 1 Very Dissatisfied through 5 Very Satisfied | `study_material_satisfaction` |

## Proposed Output

The future interface would display **Student Dropout Risk** as Low Risk, Moderate Risk, or High Risk, optionally with probabilities for each class.

It must display this qualification prominently:

> These probabilities estimate self-reported dropout-risk categories from the training survey. They do not predict confirmed future withdrawal with certainty.

The form contains no Portuguese economic indicators, foreign admission-system variables, UK learning-platform terminology, raw identifiers, Timestamp, target responses, or the excluded academic-overwhelm question.
