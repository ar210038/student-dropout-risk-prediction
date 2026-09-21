"""Streamlit interface for the saved early-dropout risk pipeline.

The app only performs inference. Model training and preprocessing fitting live in
``src/train_model.py`` and are never run by Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "dropout_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


MARITAL_STATUS = {
    1: "Single", 2: "Married", 3: "Widowed", 4: "Divorced",
    5: "De facto union", 6: "Legally separated",
}

APPLICATION_MODE = {
    1: "1st phase — general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase — special contingent (Azores)",
    7: "Holders of other higher-education courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase — special contingent (Madeira)",
    17: "2nd phase — general contingent",
    18: "3rd phase — general contingent",
    26: "Ordinance No. 533-A/99, item b2 (different plan)",
    27: "Ordinance No. 533-A/99, item b3 (other institution)",
    39: "Over 23 years old", 42: "Transfer", 43: "Change of course",
    44: "Technological specialization diploma holder",
    51: "Change of institution/course", 53: "Short-cycle diploma holder",
    57: "Change of institution/course (international)",
}

COURSE = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy",
    9070: "Communication Design", 9085: "Veterinary Nursing",
    9119: "Informatics Engineering", 9130: "Equinculture", 9147: "Management",
    9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education",
    9991: "Management (evening)",
}

PREVIOUS_QUALIFICATION = {
    1: "Secondary education", 2: "Higher education — bachelor's degree",
    3: "Higher education — degree", 4: "Higher education — master's degree",
    5: "Higher education — doctorate", 6: "Attended higher education",
    9: "12th year of schooling — not completed",
    10: "11th year of schooling — not completed", 12: "Other — 11th year of schooling",
    14: "10th year of schooling", 15: "10th year of schooling — not completed",
    19: "Basic education, 3rd cycle", 38: "Basic education, 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education — degree (1st cycle)",
    42: "Professional higher technical course",
    43: "Higher education — master's (2nd cycle)",
}

NATIONALITY = {
    1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch",
    14: "English", 17: "Lithuanian", 21: "Angolan", 22: "Cape Verdean",
    24: "Guinean", 25: "Mozambican", 26: "Santomean", 32: "Turkish",
    41: "Brazilian", 62: "Romanian", 100: "Moldovan", 101: "Mexican",
    103: "Ukrainian", 105: "Russian", 108: "Cuban", 109: "Colombian",
}

PARENT_QUALIFICATION = {
    1: "Secondary education — 12th year", 2: "Higher education — bachelor's degree",
    3: "Higher education — degree", 4: "Higher education — master's degree",
    5: "Higher education — doctorate", 6: "Attended higher education",
    9: "12th year — not completed", 10: "11th year — not completed",
    11: "7th year (old system)", 12: "Other — 11th year",
    13: "2nd year complementary high school", 14: "10th year",
    18: "General commerce course", 19: "Basic education, 3rd cycle",
    20: "Complementary high school course", 22: "Technical-professional course",
    25: "Complementary high school — not completed", 26: "7th year of schooling",
    27: "2nd cycle general high school", 29: "9th year — not completed",
    30: "8th year of schooling", 31: "General administration and commerce course",
    33: "Supplementary accounting and administration", 34: "Unknown",
    35: "Cannot read or write", 36: "Can read without completing 4th year",
    37: "Basic education, 1st cycle", 38: "Basic education, 2nd cycle",
    39: "Technological specialization course",
    40: "Higher education — degree (1st cycle)", 41: "Specialized higher studies",
    42: "Professional higher technical course",
    43: "Higher education — master's (2nd cycle)",
    44: "Higher education — doctorate (3rd cycle)",
}

PARENT_OCCUPATION = {
    0: "Student", 1: "Legislative/executive representatives, directors and managers",
    2: "Intellectual and scientific specialists",
    3: "Intermediate-level technicians and professions", 4: "Administrative staff",
    5: "Personal services, security and sales workers",
    6: "Skilled agriculture, fisheries and forestry workers",
    7: "Skilled industry, construction and craft workers",
    8: "Machine operators and assembly workers", 9: "Unskilled workers",
    10: "Armed Forces", 90: "Other situation", 99: "Unspecified",
    101: "Armed Forces officers", 102: "Armed Forces sergeants",
    103: "Other Armed Forces personnel",
    112: "Administrative and commercial services directors",
    114: "Hotel, catering, trade and other service directors",
    121: "Physical sciences, mathematics and engineering specialists",
    122: "Health professionals", 123: "Teachers",
    124: "Finance, administration and commercial specialists",
    125: "Information and communication technology specialists",
    131: "Intermediate science and engineering technicians",
    132: "Intermediate health technicians",
    134: "Intermediate legal, social, cultural and sports services",
    135: "Information and communication technology technicians",
    141: "Office, secretarial and data-processing workers",
    143: "Data, accounting, statistics and financial operators",
    144: "Other administrative support staff", 151: "Personal service workers",
    152: "Sellers", 153: "Personal care workers",
    154: "Protection and security personnel",
    161: "Market-oriented farmers and skilled agricultural workers",
    163: "Subsistence farmers, fishers, hunters and gatherers",
    171: "Skilled construction workers (except electricians)",
    172: "Skilled metallurgy and metalworking workers",
    173: "Printing, precision, jewellery and craft workers",
    174: "Skilled electrical and electronics workers",
    175: "Food, wood, clothing and other craft workers",
    181: "Fixed-plant and machine operators", 182: "Assembly workers",
    183: "Vehicle drivers and mobile-equipment operators", 191: "Cleaning workers",
    192: "Unskilled agriculture, fisheries and forestry workers",
    193: "Unskilled extractive, construction, manufacturing and transport workers",
    194: "Meal preparation assistants", 195: "Street vendors and service providers",
}

BINARY = {0: "No", 1: "Yes"}
GENDER = {0: "Female", 1: "Male"}
ATTENDANCE = {0: "Evening", 1: "Daytime"}

CATEGORY_MAPPINGS = {
    "Marital status": MARITAL_STATUS, "Application mode": APPLICATION_MODE,
    "Course": COURSE, "Daytime/evening attendance": ATTENDANCE,
    "Previous qualification": PREVIOUS_QUALIFICATION, "Nacionality": NATIONALITY,
    "Mother's qualification": PARENT_QUALIFICATION,
    "Father's qualification": PARENT_QUALIFICATION,
    "Mother's occupation": PARENT_OCCUPATION, "Father's occupation": PARENT_OCCUPATION,
    "Displaced": BINARY, "Educational special needs": BINARY, "Debtor": BINARY,
    "Tuition fees up to date": BINARY, "Gender": GENDER,
    "Scholarship holder": BINARY, "International": BINARY,
}

OBSERVED_CODES = {
    "Marital status": [1, 2, 3, 4, 5, 6],
    "Application mode": [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57],
    "Course": [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991],
    "Daytime/evening attendance": [0, 1],
    "Previous qualification": [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
    "Nacionality": [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109],
    "Mother's qualification": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    "Father's qualification": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    "Mother's occupation": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194],
    "Father's occupation": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195],
    "Displaced": [0, 1], "Educational special needs": [0, 1], "Debtor": [0, 1],
    "Tuition fees up to date": [0, 1], "Gender": [0, 1],
    "Scholarship holder": [0, 1], "International": [0, 1],
}

DEFAULT_CODES = {
    "Marital status": 1, "Application mode": 1, "Course": 9500,
    "Daytime/evening attendance": 1, "Previous qualification": 1,
    "Nacionality": 1, "Mother's qualification": 1, "Father's qualification": 37,
    "Mother's occupation": 9, "Father's occupation": 9, "Displaced": 1,
    "Educational special needs": 0, "Debtor": 0, "Tuition fees up to date": 1,
    "Gender": 0, "Scholarship holder": 0, "International": 0,
}


@st.cache_resource
def load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load the fitted pipeline and its metadata once per Streamlit process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return joblib.load(MODEL_PATH), metadata


def risk_level(probability: float) -> str:
    """Return presentation bands; 0.40 remains the binary decision threshold."""
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.40:
        return "Moderate Risk"
    return "High Risk"


def make_input_frame(values: dict[str, int | float], features: list[str]) -> pd.DataFrame:
    """Create one inference row in the exact saved training-column order."""
    missing = [feature for feature in features if feature not in values]
    extra = [feature for feature in values if feature not in features]
    if missing or extra:
        raise ValueError(f"Input mismatch. Missing={missing}; unexpected={extra}")
    return pd.DataFrame([{feature: values[feature] for feature in features}], columns=features)


def dropout_probability(pipeline: Any, row: pd.DataFrame) -> float:
    """Read the probability column belonging to the positive Dropout=1 class."""
    class_positions = np.flatnonzero(np.asarray(pipeline.classes_) == 1)
    if len(class_positions) != 1:
        raise ValueError("Saved pipeline does not contain exactly one Dropout=1 class.")
    return float(pipeline.predict_proba(row)[0, int(class_positions[0])])


def individual_contributions(
    pipeline: Any, row: pd.DataFrame, metadata: dict[str, Any]
) -> pd.DataFrame:
    """Aggregate encoded coefficient contributions back to the 24 source fields."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    encoded = preprocessor.transform(row)
    encoded_values = encoded.toarray().ravel() if hasattr(encoded, "toarray") else np.asarray(encoded).ravel()
    coefficients = np.asarray(estimator.coef_[0], dtype=float)
    if encoded_values.shape != coefficients.shape:
        raise ValueError("Transformed input and coefficient lengths do not match.")

    encoder = preprocessor.named_transformers_["categorical"]
    rows: list[dict[str, Any]] = []
    offset = 0
    for feature, categories in zip(metadata["categorical_features"], encoder.categories_, strict=True):
        width = len(categories)
        contribution = float(np.dot(encoded_values[offset:offset + width], coefficients[offset:offset + width]))
        rows.append({"Feature": feature, "Contribution": contribution})
        offset += width
    for feature in metadata["numeric_features"]:
        rows.append({"Feature": feature, "Contribution": float(encoded_values[offset] * coefficients[offset])})
        offset += 1
    if offset != len(coefficients):
        raise ValueError("Not all fitted coefficients were assigned to an input feature.")

    result = pd.DataFrame(rows)
    result["Direction"] = np.where(
        result["Contribution"] >= 0,
        "Toward higher estimated risk", "Toward lower estimated risk",
    )
    result["Strength"] = result["Contribution"].abs()
    return result.sort_values("Strength", ascending=False).head(5).reset_index(drop=True)


def category_input(feature: str, label: str, help_text: str | None = None) -> int:
    codes = OBSERVED_CODES[feature]
    mapping = CATEGORY_MAPPINGS[feature]
    return int(st.selectbox(
        label, options=codes, index=codes.index(DEFAULT_CODES[feature]),
        format_func=lambda code: mapping.get(code, f"Category {code}"),
        help=help_text,
    ))


def render_home(metadata: dict[str, Any]) -> None:
    st.title("Early Student Dropout Risk Prediction Using Machine Learning")
    st.write(
        "This system estimates the probability that a newly enrolled university student "
        "may later drop out, using demographic, admission, financial, and contextual "
        "information available at enrollment."
    )
    st.markdown(
        '<div class="workflow">Student Information <span>→</span> ML Model <span>→</span> '
        'Dropout Probability <span>→</span> Risk Level <span>→</span> Early Support</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    col1.metric("Resolved cohort", "3,630 students")
    col2.metric("Model", metadata["model_name"])
    col1, col2 = st.columns(2)
    col1.metric("Prediction point", "At enrollment")
    col2.metric("Purpose", "Early academic support")
    st.markdown("#### Dataset")
    st.write("UCI Predict Students' Dropout and Academic Success")
    st.warning(
        "This model estimates statistical risk and should not be used as the sole basis "
        "for academic, financial, disciplinary, or admission decisions."
    )


def render_prediction(pipeline: Any, metadata: dict[str, Any]) -> None:
    st.title("Predict Dropout Risk")
    st.caption(
        "Enter information available at enrollment. Category names are converted to the "
        "official UCI codes expected by the saved pipeline."
    )
    with st.form("prediction_form"):
        st.subheader("Student background")
        left, right = st.columns(2)
        with left:
            marital = category_input("Marital status", "Marital status")
            age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=20, step=1)
            gender = category_input("Gender", "Gender")
            nationality = category_input("Nacionality", "Nationality")
        with right:
            international = category_input("International", "International student")
            displaced = category_input(
                "Displaced", "Relocated for study",
                "Whether the student moved away from their usual residence to study.",
            )
            special_needs = category_input(
                "Educational special needs", "Educational special needs",
                "Whether educational special-needs support was recorded at enrollment.",
            )

        st.subheader("Admission information")
        left, right = st.columns(2)
        with left:
            app_mode = category_input(
                "Application mode", "Application route",
                "The documented admission route or application scheme used by the student.",
            )
            app_order = st.number_input(
                "Programme preference order", min_value=0, max_value=9, value=1, step=1,
                help="The programme's order in the application preferences: 0 is first choice and 9 is last choice.",
            )
            course = category_input("Course", "Degree programme")
            attendance = category_input("Daytime/evening attendance", "Attendance schedule")
        with right:
            previous = category_input(
                "Previous qualification", "Previous qualification",
                "The highest qualification recorded before university admission.",
            )
            previous_grade = st.number_input(
                "Previous qualification grade", min_value=95.0, max_value=190.0,
                value=133.1, step=0.1,
                help="Grade recorded for the previous qualification; observed dataset range: 95–190.",
            )
            admission_grade = st.number_input(
                "Admission grade", min_value=95.0, max_value=190.0,
                value=126.1, step=0.1,
                help="University admission grade; observed dataset range: 95–190.",
            )

        st.subheader("Family background")
        left, right = st.columns(2)
        with left:
            mother_qualification = category_input(
                "Mother's qualification", "Mother's qualification",
                "Select the closest category documented in the source dataset.",
            )
            mother_occupation = category_input(
                "Mother's occupation", "Mother's occupation",
                "Select the closest occupational group documented in the source dataset.",
            )
        with right:
            father_qualification = category_input(
                "Father's qualification", "Father's qualification",
                "Select the closest category documented in the source dataset.",
            )
            father_occupation = category_input(
                "Father's occupation", "Father's occupation",
                "Select the closest occupational group documented in the source dataset.",
            )

        st.subheader("Financial and support information")
        left, right = st.columns(2)
        with left:
            debtor = category_input(
                "Debtor", "Outstanding debt recorded",
                "Whether the institution recorded the student as a debtor at enrollment.",
            )
            tuition = category_input(
                "Tuition fees up to date", "Tuition fees up to date",
                "Whether tuition payments were current at the prediction point.",
            )
        with right:
            scholarship = category_input("Scholarship holder", "Scholarship holder")

        st.subheader("Economic context")
        left, right = st.columns(2)
        with left:
            unemployment = st.number_input(
                "Unemployment rate (%)", min_value=7.6, max_value=16.2, value=11.1, step=0.1,
                help="National rate recorded for the student's enrollment period; observed range: 7.6–16.2%.",
            )
            inflation = st.number_input(
                "Inflation rate (%)", min_value=-0.8, max_value=3.7, value=1.4, step=0.1,
                help="National rate recorded for the student's enrollment period; observed range: −0.8–3.7%.",
            )
        with right:
            gdp = st.number_input(
                "GDP growth rate (%)", min_value=-4.06, max_value=3.51, value=0.32, step=0.01,
                help="National GDP growth recorded for the student's enrollment period; observed range: −4.06–3.51%.",
            )

        submitted = st.form_submit_button("Predict Dropout Risk", type="primary", width="stretch")

    if not submitted:
        return

    values: dict[str, int | float] = {
        "Marital status": marital, "Application mode": app_mode,
        "Application order": int(app_order), "Course": course,
        "Daytime/evening attendance": attendance, "Previous qualification": previous,
        "Previous qualification (grade)": float(previous_grade), "Nacionality": nationality,
        "Mother's qualification": mother_qualification, "Father's qualification": father_qualification,
        "Mother's occupation": mother_occupation, "Father's occupation": father_occupation,
        "Admission grade": float(admission_grade), "Displaced": displaced,
        "Educational special needs": special_needs, "Debtor": debtor,
        "Tuition fees up to date": tuition, "Gender": gender,
        "Scholarship holder": scholarship, "Age at enrollment": int(age),
        "International": international, "Unemployment rate": float(unemployment),
        "Inflation rate": float(inflation), "GDP": float(gdp),
    }
    try:
        row = make_input_frame(values, metadata["feature_list"])
        probability = dropout_probability(pipeline, row)
    except Exception as exc:
        st.error(f"Prediction could not be calculated: {exc}")
        return

    level = risk_level(probability)
    color_class = {"Low Risk": "risk-low", "Moderate Risk": "risk-moderate", "High Risk": "risk-high"}[level]
    st.divider()
    st.subheader("Dropout Probability")
    st.markdown(
        f'<div class="result-card {color_class}"><div class="risk-label">{level}</div>'
        f'<div class="probability">{probability:.1%}</div>'
        f'<div>Predicted dropout probability</div></div>', unsafe_allow_html=True,
    )
    st.progress(probability)
    threshold = float(metadata["threshold"])
    st.caption(
        f"The model's classification threshold is {threshold:.0%}. The Low/Moderate/High labels "
        "are presentation categories, not separately trained ML classes."
    )
    st.info(
        "Students with similar enrollment profiles were associated with this estimated likelihood "
        "of dropout in the training dataset. Use the result to support further academic review, "
        "not as a final decision."
    )

    st.subheader("Factors associated with the prediction")
    try:
        factors = individual_contributions(pipeline, row, metadata)
        display = factors[["Feature", "Direction", "Contribution"]].copy()
        display["Contribution"] = display["Contribution"].map(lambda value: f"{value:+.3f}")
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption(
            "These are the five largest contributions to the model's log-odds for this input. "
            "They describe fitted associations and do not establish that any feature causes dropout."
        )
    except Exception:
        st.caption("Individual contribution details are unavailable for this saved model structure.")
        figure = FIGURES_DIR / "feature_importance.png"
        if figure.exists():
            st.image(str(figure), caption="Global model feature importance", width="stretch")


def render_performance(metadata: dict[str, Any]) -> None:
    st.title("Model Performance")
    metrics = metadata["held_out_test_metrics"]
    st.write(f"**Final model:** {metadata['model_name']}")
    st.write(f"**Prediction threshold:** {float(metadata['threshold']):.2f}")
    cols = st.columns(5)
    metric_pairs = [
        ("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
        ("F1-score", "f1"), ("ROC-AUC", "roc_auc"),
    ]
    for column, (label, key) in zip(cols, metric_pairs, strict=True):
        column.metric(label, f"{float(metrics[key]):.2%}")
    st.info(
        "Recall was prioritized because the goal is early identification of students who may need support. "
        "The model correctly identified about 83% of dropout cases in the held-out test set; this is not overall accuracy."
    )

    st.subheader("Held-out confusion matrix")
    col1, col2 = st.columns(2)
    col1.metric("Correctly detected dropouts", int(metrics["tp"]))
    col1.metric("Missed dropouts", int(metrics["fn"]))
    col2.metric("False dropout warnings", int(metrics["fp"]))
    col2.metric("Correctly identified graduates", int(metrics["tn"]))

    first_row = st.columns(2)
    figures = [
        ("confusion_matrix.png", "Confusion matrix at the saved 0.40 threshold", first_row[0]),
        ("roc_curve.png", "Receiver operating characteristic curve", first_row[1]),
        ("feature_importance.png", "Global feature importance / coefficient magnitude", st.container()),
    ]
    for filename, caption, container in figures:
        path = FIGURES_DIR / filename
        with container:
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")
            else:
                st.warning(f"Optional performance figure is unavailable: {path.name}")


def render_dataset() -> None:
    st.title("About the Dataset")
    st.markdown(
        "The project uses the [UCI Predict Students' Dropout and Academic Success dataset]"
        "(https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). "
        "It is distributed under the **CC BY 4.0** license."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Original students", "4,424")
    col2.metric("Resolved cohort", "3,630")
    col3.metric("Prediction point", "At enrollment")
    st.markdown(
        """
        #### Outcome definition

        - Graduate: 2,209 students
        - Dropout: 1,421 students
        - Enrolled: 794 students

        Students still labelled **Enrolled** were excluded because their final outcome is unresolved.
        The binary target is **Dropout = 1** and **Graduate = 0**.

        #### Leakage-aware feature timing

        Prediction is made at enrollment, before first-semester teaching begins. All first- and
        second-semester curricular-unit variables were intentionally excluded because they would not
        be available at this point and the dataset does not give exact dropout dates.

        #### Limitations

        - The data comes from one Portuguese higher-education institution.
        - Results may not generalize to Bangladesh or other universities.
        - The dataset covers multiple degree programs, not only Computer Science.
        - Exact dropout dates are not provided.
        - Sensitive demographic and socioeconomic variables create fairness concerns.
        - Predictions reflect statistical association, not causation.
        - The model must not be the sole basis for high-stakes decisions.
        """
    )


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3 {color: #123b66;}
        .workflow {background: #eef5fb; border: 1px solid #caddec; border-radius: 10px;
                   padding: 1.1rem; margin: 1.4rem 0; text-align: center; font-weight: 600; color: #193b5d;}
        .workflow span {color: #2474b8; padding: 0 .45rem;}
        .result-card {border-left: 6px solid; border-radius: 10px; padding: 1.25rem 1.5rem;
                      margin-bottom: 1rem; background: #f8fafc;}
        .risk-low {border-color: #2e7d32;} .risk-moderate {border-color: #ed9b18;}
        .risk-high {border-color: #bd3039;}
        .risk-label {font-size: 1.25rem; font-weight: 700; color: #183b5c;}
        .probability {font-size: 2.25rem; font-weight: 750; color: #123b66; line-height: 1.2;}
        [data-testid="stSidebar"] {background-color: #f4f8fc;}
        </style>
        """, unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Early Dropout Risk", page_icon="ED", layout="wide")
    apply_style()
    st.sidebar.title("Project Navigation")
    page = st.sidebar.radio(
        "Go to", ["Home", "Predict Dropout Risk", "Model Performance", "About Dataset"],
        label_visibility="collapsed",
    )
    st.sidebar.caption("Local academic decision-support prototype")

    try:
        pipeline, metadata = load_artifacts()
    except Exception as exc:
        st.error(
            "The saved inference artifacts could not be loaded. Run `python src/train_model.py` "
            f"once, then restart Streamlit. Details: {exc}"
        )
        st.stop()

    if page == "Home":
        render_home(metadata)
    elif page == "Predict Dropout Risk":
        render_prediction(pipeline, metadata)
    elif page == "Model Performance":
        render_performance(metadata)
    else:
        render_dataset()


if __name__ == "__main__":
    main()
