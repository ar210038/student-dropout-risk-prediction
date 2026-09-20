"""Streamlit interface for Bangladesh student profiling and factor analysis."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "bangladesh_student_profiler.joblib"
METADATA_PATH = ROOT / "models" / "bangladesh_student_profiler_metadata.json"
PROFILE_PATH = ROOT / "reports" / "bangladesh_risk" / "profile_definitions.json"
PCA_PATH = ROOT / "reports" / "bangladesh_risk" / "figures" / "cluster_pca.png"
PROFILE_FEATURES = ["age_group", "academic_level", "personal_income", "employment_type", "internet_quality", "living_arrangement", "study_routine", "family_income", "scholarship", "study_space_quality", "academic_resource_access", "current_gpa"]
INPUT_OPTIONS: dict[str, list[Any]] = {
    "age_group": ["17–20", "21–23", "24+"],
    "academic_level": ["Undergraduate (Year 1)", "Undergraduate (Year 2–4)", "Masters"],
    "personal_income": ["No personal income", "<5,000 BDT", "5,000–10,000 BDT", ">10,000 BDT"],
    "employment_type": ["No", "Yes, Tuition/Coaching", "Yes, Freelancing (e.g., online work)", "Yes, Part-time/Full-time Job"],
    "internet_quality": [1, 2, 3, 4, 5],
    "living_arrangement": ["Home or Hostels", "University Residence (Halls)"],
    "study_routine": ["Only before exams", "Sporadically (no fixed schedule)", "Regularly throughout the course (e.g., weekly)"],
    "family_income": ["<20,000", "20,000–50,000", "50,001–100,000", ">100,000"],
    "scholarship": ["No", "Yes, Government Scholarship", "Yes, University Merit-Based", "Yes, Private/Donor-Funded"],
    "study_space_quality": ["No dedicated study space", "Yes - Very noisy (1)", "Yes - Noisy (2)", "Yes - Moderate noise (3)", "Yes - Quiet (4)", "Yes - Very quiet (5)"],
    "academic_resource_access": ["No access", "Yes, Rarely", "Yes, Weekly", "Yes, Daily"],
    "current_gpa": ["<2.5", "2.5–3.0", "3.1–3.5", ">3.5"],
}
INPUT_LABELS = {"age_group": "Age Group", "academic_level": "Academic Level", "personal_income": "Personal Income", "employment_type": "Employment / Tuition Work", "internet_quality": "Internet Quality", "living_arrangement": "Living Arrangement", "study_routine": "Study Routine", "family_income": "Monthly Family Income (BDT)", "scholarship": "Scholarship / Stipend", "study_space_quality": "Study-Space Quality", "academic_resource_access": "Academic-Resource Access", "current_gpa": "Current GPA"}
PROFILE_NAMES = {0: "Profile 1 — Early-Stage, Mostly Non-Working Students", 1: "Profile 2 — Advanced-Stage, Working Students"}
FACTOR_ASSOCIATIONS = pd.DataFrame({"Factor": ["Study-material satisfaction", "Scholarship / stipend", "Study-space quality", "Employment type", "Prior residence", "Sleep duration"], "Cramér’s V": [0.185, 0.171, 0.170, 0.169, 0.167, 0.153]})

@st.cache_resource
def load_profiler() -> Any:
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_project_data() -> tuple[dict, list[dict]]:
    return (json.loads(METADATA_PATH.read_text(encoding="utf-8")), json.loads(PROFILE_PATH.read_text(encoding="utf-8")))

def make_profile_input(values: dict[str, Any]) -> pd.DataFrame:
    if set(values) != set(PROFILE_FEATURES):
        raise ValueError("Profile answers do not match the required 12-input schema.")
    return pd.DataFrame([{feature: values[feature] for feature in PROFILE_FEATURES}])

def assign_profile(profiler: Any, values: dict[str, Any]) -> int:
    cluster_id = int(profiler.predict(make_profile_input(values))[0])
    if cluster_id not in PROFILE_NAMES:
        raise ValueError(f"Unexpected profile ID: {cluster_id}")
    return cluster_id

def profile_characteristics(profile: dict, limit: int = 5) -> list[str]:
    preferred = ["age_group", "academic_level", "employment_type", "personal_income", "study_routine"]
    return [f"{INPUT_LABELS[f]}: {profile['dominant_values'][f]['value']} ({profile['dominant_values'][f]['percentage']:.1f}% of this profile)" for f in preferred[:limit]]

def metric_cards(items: list[tuple[str, str]]) -> None:
    for column, (label, value) in zip(st.columns(len(items)), items):
        column.metric(label, value)

def page_home() -> None:
    st.title("Student Profiling and Dropout Risk Factor Analysis")
    st.subheader("Machine Learning-Based Analysis of Bangladeshi University Students")
    st.write("This academic prototype uses unsupervised machine learning to identify common student profiles from academic, socioeconomic and lifestyle characteristics. It also explores how selected factors relate to students’ self-reported consideration of dropping out due to academic stress.")
    st.markdown('<div class="workflow"><span>Student Survey Information</span><b>→</b><span>Data Preprocessing</span><b>→</b><span>K-Means Profiling</span><b>→</b><span>Student Profile</span><strong>+</strong><span>Dropout Risk Factor Analysis</span></div>', unsafe_allow_html=True)
    metric_cards([("Dataset", "Bangladesh survey"), ("Raw responses", "368"), ("Cleaned responses", "351"), ("ML method", "K-Means"), ("Profiles", "2")])
    st.info("The system identifies descriptive patterns and explores associations. It does not forecast an individual student’s future outcome.")

def page_profile_explorer(profiler: Any, profiles: list[dict]) -> None:
    st.title("Student Profile Explorer")
    st.write("Enter 12 academic, socioeconomic and lifestyle characteristics to find the closest survey profile.")
    with st.form("profile_form"):
        left, right = st.columns(2)
        answers: dict[str, Any] = {}
        for index, feature in enumerate(PROFILE_FEATURES):
            with (left if index % 2 == 0 else right):
                if feature == "internet_quality":
                    answers[feature] = st.select_slider(INPUT_LABELS[feature], options=INPUT_OPTIONS[feature], value=3, help="1 = very poor, 5 = very good")
                else:
                    answers[feature] = st.selectbox(INPUT_LABELS[feature], INPUT_OPTIONS[feature])
        submitted = st.form_submit_button("Find Student Profile", type="primary", width="stretch")
    if submitted:
        cluster_id = assign_profile(profiler, answers)
        profile = next(item for item in profiles if item["cluster_id"] == cluster_id)
        number, name = PROFILE_NAMES[cluster_id].split(" — ", maxsplit=1)
        st.markdown("### Your Student Profile")
        st.markdown(f'<div class="result-card"><div class="eyebrow">{number}</div><h2>{name}</h2><p>This profile represents the group in the survey dataset whose characteristics are most similar to the responses entered above.</p></div>', unsafe_allow_html=True)
        st.markdown("#### Defining characteristics in the survey")
        for description in profile_characteristics(profile):
            st.markdown(f"- {description}")
        st.warning("Profile assignment is descriptive and is not a prediction of whether a student will drop out.")
    st.markdown("### Profile comparison")
    metric_cards([("Profile 1", "213 students · 60.7%"), ("Profile 2", "138 students · 39.3%")])

def page_factor_analysis() -> None:
    st.title("Dropout Risk Factor Analysis")
    st.write("The survey asked students how likely they were to consider dropping out due to academic stress. This section explores statistical associations between student characteristics and that self-reported response.")
    metric_cards([("Not Elevated (responses 1–3)", "274 · 78.1%"), ("Elevated (responses 4–5)", "77 · 21.9%")])
    st.caption("This binary grouping is used only for descriptive analysis.")
    st.markdown("### Association with Elevated Dropout Consideration")
    chart_data = FACTOR_ASSOCIATIONS.sort_values("Cramér’s V")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(chart_data["Factor"], chart_data["Cramér’s V"], color="#277da1")
    ax.set_xlabel("Cramér’s V"); ax.set_xlim(0, 0.21); ax.spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(chart_data["Cramér’s V"]): ax.text(value + 0.003, index, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout(); st.pyplot(fig, width="stretch"); plt.close(fig)
    st.caption("Higher values indicate stronger association within this dataset. These results do not establish causation.")
    st.markdown("### Student profiles and dropout consideration")
    comparison = pd.DataFrame({"Profile": ["Profile 1", "Profile 2"], "Elevated (%)": [20.7, 23.9]})
    st.bar_chart(comparison.set_index("Profile"), color="#6c8ebf", horizontal=True)
    st.error("The difference was not statistically significant.")
    metric_cards([("χ²(1)", "0.346"), ("p-value", "0.557"), ("Cramér’s V", "0.038")])
    st.write("Student-profile membership had negligible association with elevated dropout consideration in this sample.")

def page_methodology(metadata: dict) -> None:
    st.title("Methodology & Dataset")
    st.markdown("### Dataset and cleaning")
    metric_cards([("Original responses", "368"), ("Duplicate copies removed", "17"), ("Unique profiles", "351"), ("Clustering features", "12")])
    st.write("Two duplicate-profile groups were detected. One contained 16 near-immediate repeated submissions and another contained 3 repeated submissions. Duplicate copies were removed without exposing respondent information.")
    st.markdown("- Target excluded from clustering: **Yes**\n- Timestamp excluded: **Yes**\n- Academic-overwhelm question excluded from primary clustering: **Yes**")
    st.markdown("### Unsupervised machine learning")
    st.write("K-Means is an unsupervised machine-learning algorithm that groups observations according to similarity. Unlike supervised classification, it does not require a predefined target label during training. The value k = 2 was selected after evaluating k = 2–5.")
    metric_cards([("Silhouette", f"{metadata['silhouette']:.4f}"), ("Davies–Bouldin", f"{metadata['davies_bouldin']:.4f}"), ("Mean ARI stability", f"{metadata['stability_pairwise_ari_mean']:.4f}")])
    st.write("The relatively low silhouette score indicates that the profiles are not sharply separated. The high stability score indicates that repeated clustering runs produced almost the same grouping. The profiles were highly reproducible but only weakly separated geometrically.")
    st.image(str(PCA_PATH), caption="Two-dimensional PCA visualization of the student profiles. PCA is shown for visualization only and was not used as a dropout-risk target.")
    st.markdown("### Limitations")
    st.markdown("- Small, self-reported survey sample with possible selection bias\n- Not nationally representative and no confirmed future dropout outcome\n- Cross-sectional rather than longitudinal data\n- Broad descriptive profiles with weak silhouette separation\n- Profile membership was not significantly associated with dropout consideration\n- Factor associations are exploratory; association does not imply causation")

def apply_style() -> None:
    st.markdown("""<style>
    .block-container {max-width:1120px;padding-top:2.2rem;padding-bottom:3rem} h1{color:#17324d;letter-spacing:-.025em} h2,h3{color:#244d68}
    div[data-testid="stMetric"]{background:#f4f8fb;border:1px solid #dbe7ef;border-radius:14px;padding:1rem}
    .workflow{display:flex;flex-wrap:wrap;gap:.7rem;align-items:center;margin:1.5rem 0 2rem}.workflow span{background:#eef6f8;color:#174b5e;border:1px solid #cfe2e8;padding:.65rem .8rem;border-radius:10px;font-weight:600}.workflow strong{font-size:1.3rem;color:#277da1}
    .result-card{background:linear-gradient(135deg,#edf7f7,#f5f8fc);border:1px solid #cfe2e8;border-left:6px solid #277da1;border-radius:16px;padding:1.3rem 1.5rem;margin:.5rem 0 1rem}.result-card h2{margin:.15rem 0 .5rem}.eyebrow{text-transform:uppercase;letter-spacing:.08em;color:#277da1;font-weight:750}
    @media(max-width:700px){.workflow b{display:none}div[data-testid="stHorizontalBlock"]{gap:.5rem}}</style>""", unsafe_allow_html=True)

def main() -> None:
    st.set_page_config(page_title="Bangladesh Student Profiling", page_icon="🎓", layout="wide")
    apply_style(); metadata, profiles = load_project_data(); profiler = load_profiler()
    st.sidebar.title("Student Profiling")
    page = st.sidebar.radio("Navigate", ["Home", "Student Profile Explorer", "Dropout Risk Factor Analysis", "Methodology & Dataset"])
    st.sidebar.caption("Academic prototype · Bangladesh university survey")
    if page == "Home": page_home()
    elif page == "Student Profile Explorer": page_profile_explorer(profiler, profiles)
    elif page == "Dropout Risk Factor Analysis": page_factor_analysis()
    else: page_methodology(metadata)

if __name__ == "__main__": main()
