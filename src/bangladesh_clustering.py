"""Unsupervised student profiling for the cleaned Bangladesh survey."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

try:
    from .bangladesh_risk_pipeline import (
        PRIMARY_CANDIDATE_FEATURES,
        elevated_risk,
        load_and_clean,
        primary_model_frame,
    )
except ImportError:  # Support direct execution: python src/bangladesh_clustering.py
    from bangladesh_risk_pipeline import (
        PRIMARY_CANDIDATE_FEATURES,
        elevated_risk,
        load_and_clean,
        primary_model_frame,
    )


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "bangladesh_risk" / "raw" / "Student Dropout Risk Survey Dataset.xlsx"
REPORT_DIR = ROOT / "reports" / "bangladesh_risk"
FIGURE_DIR = REPORT_DIR / "figures"
RESULTS_PATH = REPORT_DIR / "clustering_results.json"
PROFILE_PATH = REPORT_DIR / "profile_definitions.json"
MODEL_PATH = ROOT / "models" / "bangladesh_student_profiler.joblib"
METADATA_PATH = ROOT / "models" / "bangladesh_student_profiler_metadata.json"
SEEDS = [11, 23, 42, 71, 101, 137, 173, 211, 257, 307]

NOMINAL_FEATURES = [
    "gender",
    "university_type",
    "prior_residence",
    "living_arrangement",
    "employment_type",
    "scholarship",
]
ORDINAL_CATEGORIES = {
    "age_group": ["17–20", "21–23", "24+"],
    "academic_level": [
        "Undergraduate (Year 1)",
        "Undergraduate (Year 2–4)",
        "Masters",
    ],
    "sleep_duration": ["<5 hours", "5–6 hours", "7–8 hours", ">8 hours"],
    "meal_skipping": ["Never", "1–2 times/week", "3–4 times/week", "Daily"],
    "commute_time": ["<30 minutes", "30–60 minutes", ">1 hour"],
    "study_space_quality": [
        "No dedicated study space",
        "Yes - Very noisy (1)",
        "Yes - Noisy (2)",
        "Yes - Moderate noise (3)",
        "Yes - Quiet (4)",
        "Yes - Very quiet (5)",
    ],
    "current_gpa": ["<2.5", "2.5–3.0", "3.1–3.5", ">3.5"],
    "study_routine": [
        "Only before exams",
        "Sporadically (no fixed schedule)",
        "Regularly throughout the course (e.g., weekly)",
    ],
    "attendance": ["<50%", "50–75%", ">75%"],
    "family_income": ["<20,000", "20,000–50,000", "50,001–100,000", ">100,000"],
    "personal_income": [
        "No personal income",
        "<5,000 BDT",
        "5,000–10,000 BDT",
        ">10,000 BDT",
    ],
    "academic_resource_access": ["No access", "Yes, Rarely", "Yes, Weekly", "Yes, Daily"],
}
NUMERIC_ORDINAL_FEATURES = ["internet_quality", "study_material_satisfaction"]
CLUSTER_FEATURES = PRIMARY_CANDIDATE_FEATURES.copy()
FINAL_PROFILE_FEATURES = [
    "age_group",
    "academic_level",
    "personal_income",
    "employment_type",
    "internet_quality",
    "living_arrangement",
    "study_routine",
    "family_income",
    "scholarship",
    "study_space_quality",
    "academic_resource_access",
    "current_gpa",
]
PROFILE_LABELS = {
    0: "Profile 1 — Early-Stage, Mostly Non-Working Students",
    1: "Profile 2 — Advanced-Stage, Working Students",
}


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    nominal = [feature for feature in features if feature in NOMINAL_FEATURES]
    ordinal = [feature for feature in features if feature in ORDINAL_CATEGORIES]
    numeric = [feature for feature in features if feature in NUMERIC_ORDINAL_FEATURES]
    return ColumnTransformer(
        [
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal,
            ),
            (
                "ordinal",
                Pipeline(
                    [
                        (
                            "encode",
                            OrdinalEncoder(
                                categories=[ORDINAL_CATEGORIES[feature] for feature in ordinal],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                ordinal,
            ),
            ("numeric", StandardScaler(), numeric),
        ],
        verbose_feature_names_out=True,
    )


def cramers_v(table: pd.DataFrame) -> float:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt((chi2 / n) / denominator)) if denominator else 0.0


def clustering_runs(matrix: np.ndarray, k: int) -> tuple[list[np.ndarray], dict]:
    labels = []
    silhouettes, db_scores, size_runs = [], [], []
    for seed in SEEDS:
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
        current = model.fit_predict(matrix)
        labels.append(current)
        silhouettes.append(silhouette_score(matrix, current))
        db_scores.append(davies_bouldin_score(matrix, current))
        size_runs.append(sorted(np.bincount(current, minlength=k).tolist()))
    pairwise_ari = [
        adjusted_rand_score(left, right)
        for left, right in itertools.combinations(labels, 2)
    ]
    return labels, {
        "k": k,
        "silhouette_mean": float(np.mean(silhouettes)),
        "silhouette_std": float(np.std(silhouettes, ddof=1)),
        "davies_bouldin_mean": float(np.mean(db_scores)),
        "davies_bouldin_std": float(np.std(db_scores, ddof=1)),
        "pairwise_ari_mean": float(np.mean(pairwise_ari)),
        "pairwise_ari_min": float(np.min(pairwise_ari)),
        "cluster_sizes_seed_42": sorted(
            np.bincount(labels[SEEDS.index(42)], minlength=k).tolist()
        ),
        "cluster_size_range": {
            "smallest": int(min(min(run) for run in size_runs)),
            "largest": int(max(max(run) for run in size_runs)),
        },
    }


def choose_k(evaluations: list[dict]) -> int:
    # Prefer stable, non-tiny solutions, then silhouette and compactness.
    eligible = [
        row
        for row in evaluations
        if row["pairwise_ari_mean"] >= 0.75
        and row["cluster_size_range"]["smallest"] >= 0.10 * 351
    ]
    pool = eligible or evaluations
    return max(
        pool,
        key=lambda row: (
            row["silhouette_mean"],
            -row["davies_bouldin_mean"],
            row["pairwise_ari_mean"],
        ),
    )["k"]


def feature_cluster_associations(
    data: pd.DataFrame, labels: np.ndarray, features: list[str]
) -> list[dict]:
    rows = []
    for feature in features:
        table = pd.crosstab(data[feature], labels)
        rows.append({"feature": feature, "cramers_v": cramers_v(table)})
    return sorted(rows, key=lambda row: row["cramers_v"], reverse=True)


def profile_summary(
    data: pd.DataFrame, labels: np.ndarray, features: list[str]
) -> list[dict]:
    frame = data.copy()
    frame["cluster_id"] = labels
    frame["elevated_risk"] = frame["dropout_intention_score"].map(elevated_risk)
    summaries = []
    for cluster_id, group in frame.groupby("cluster_id"):
        dominant = {}
        for feature in features:
            counts = group[feature].value_counts(normalize=True)
            dominant[feature] = {
                "value": str(counts.index[0]),
                "percentage": float(100 * counts.iloc[0]),
            }
        risk_distribution = (
            group["dropout_intention_score"]
            .value_counts()
            .reindex(range(1, 6), fill_value=0)
            .to_dict()
        )
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "profile_label": PROFILE_LABELS.get(
                    int(cluster_id), f"Profile {cluster_id + 1}"
                ),
                "size": len(group),
                "percentage": float(100 * len(group) / len(frame)),
                "dominant_values": dominant,
                "risk_mean": float(group["dropout_intention_score"].mean()),
                "risk_median": float(group["dropout_intention_score"].median()),
                "risk_distribution_1_to_5": {
                    str(key): int(value) for key, value in risk_distribution.items()
                },
                "elevated_count": int(group["elevated_risk"].sum()),
                "elevated_percentage": float(100 * group["elevated_risk"].mean()),
            }
        )
    return summaries


def individual_factor_associations(data: pd.DataFrame) -> list[dict]:
    elevated = data["dropout_intention_score"].map(elevated_risk)
    rows = []
    for feature in CLUSTER_FEATURES:
        table = pd.crosstab(data[feature], elevated)
        chi2, p_value, _, _ = chi2_contingency(table)
        category_rates = (
            pd.DataFrame({"feature": data[feature].astype(str), "elevated": elevated})
            .groupby("feature")["elevated"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
        row = {
            "feature": feature,
            "chi_square": float(chi2),
            "p_value": float(p_value),
            "cramers_v": cramers_v(table),
            "highest_elevated_category": str(category_rates.index[0]),
            "highest_elevated_percentage": float(100 * category_rates.iloc[0]["mean"]),
            "lowest_elevated_category": str(category_rates.index[-1]),
            "lowest_elevated_percentage": float(100 * category_rates.iloc[-1]["mean"]),
        }
        if feature in ORDINAL_CATEGORIES:
            mapping = {value: index for index, value in enumerate(ORDINAL_CATEGORIES[feature])}
            correlation, correlation_p = spearmanr(
                data[feature].map(mapping), data["dropout_intention_score"]
            )
            row["spearman_rho"] = float(correlation)
            row["spearman_p_value"] = float(correlation_p)
        elif feature in NUMERIC_ORDINAL_FEATURES:
            correlation, correlation_p = spearmanr(
                data[feature], data["dropout_intention_score"]
            )
            row["spearman_rho"] = float(correlation)
            row["spearman_p_value"] = float(correlation_p)
        rows.append(row)
    return sorted(rows, key=lambda row: row["cramers_v"], reverse=True)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cleaned, audit = load_and_clean(RAW_PATH)
    full_data = primary_model_frame(cleaned, CLUSTER_FEATURES)[CLUSTER_FEATURES]
    full_preprocessor = make_preprocessor(CLUSTER_FEATURES)
    full_matrix = full_preprocessor.fit_transform(full_data)
    full_evaluations = []
    for k in range(2, 6):
        _, evaluation = clustering_runs(full_matrix, k)
        full_evaluations.append(evaluation)
    full_best_k = choose_k(full_evaluations)
    full_model = KMeans(n_clusters=full_best_k, n_init=20, random_state=42)
    full_labels = full_model.fit_predict(full_matrix)
    full_cluster_association = feature_cluster_associations(
        cleaned, full_labels, CLUSTER_FEATURES
    )

    clustering_data = primary_model_frame(cleaned, FINAL_PROFILE_FEATURES)[
        FINAL_PROFILE_FEATURES
    ]
    preprocessor = make_preprocessor(FINAL_PROFILE_FEATURES)
    matrix = preprocessor.fit_transform(clustering_data)
    evaluations = []
    for k in range(2, 6):
        _, evaluation = clustering_runs(matrix, k)
        evaluations.append(evaluation)
    best_k = choose_k(evaluations)

    final_kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    labels = final_kmeans.fit_predict(matrix)
    final_silhouette = silhouette_score(matrix, labels)
    final_db = davies_bouldin_score(matrix, labels)
    final_runs, final_stability = clustering_runs(matrix, best_k)
    reference_labels = final_runs[SEEDS.index(42)]
    if adjusted_rand_score(labels, reference_labels) < 0.999:
        raise RuntimeError("Final clustering does not match the seed-42 stability run")

    profile_definitions = profile_summary(cleaned, labels, FINAL_PROFILE_FEATURES)
    cluster_association = feature_cluster_associations(
        cleaned, labels, FINAL_PROFILE_FEATURES
    )
    individual_association = individual_factor_associations(cleaned)

    risk_table = pd.crosstab(labels, cleaned["dropout_intention_score"].map(elevated_risk))
    chi2, risk_p, _, _ = chi2_contingency(risk_table)
    risk_cramers_v = cramers_v(risk_table)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(matrix)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id in range(best_k):
        mask = labels == cluster_id
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            alpha=0.70,
            s=28,
            label=f"Profile {cluster_id + 1}",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Student profiles in two-dimensional PCA view")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cluster_pca.png", dpi=180)
    plt.close(fig)

    profiler = Pipeline(
        [("preprocess", preprocessor), ("cluster", final_kmeans)]
    )
    # Components are already fitted; this object assigns future answers deterministically.
    joblib.dump(profiler, MODEL_PATH)
    metadata = {
        "project_title": "Machine Learning-Based Student Dropout Risk Profiling and Factor Analysis among Bangladeshi University Students",
        "method": "K-Means profiling with post-hoc risk interpretation",
        "sample_size": len(cleaned),
        "features": FINAL_PROFILE_FEATURES,
        "excluded": ["Timestamp", "dropout_intention_score", "risk classes", "academic_overwhelm"],
        "best_k": best_k,
        "silhouette": float(final_silhouette),
        "davies_bouldin": float(final_db),
        "stability_pairwise_ari_mean": final_stability["pairwise_ari_mean"],
        "profile_definitions": profile_definitions,
        "claim": "Descriptive student-profile assignment, not dropout prediction.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    PROFILE_PATH.write_text(json.dumps(profile_definitions, indent=2), encoding="utf-8")
    results = {
        "cleaning_audit": audit,
        "sample_size": len(cleaned),
        "initial_full_features": CLUSTER_FEATURES,
        "features": FINAL_PROFILE_FEATURES,
        "preprocessing": {
            "nominal_one_hot": [
                feature for feature in FINAL_PROFILE_FEATURES if feature in NOMINAL_FEATURES
            ],
            "ordinal_scaled": [
                feature for feature in FINAL_PROFILE_FEATURES if feature in ORDINAL_CATEGORIES
            ],
            "numeric_ratings_scaled": [
                feature
                for feature in FINAL_PROFILE_FEATURES
                if feature in NUMERIC_ORDINAL_FEATURES
            ],
        },
        "full_k_evaluations": full_evaluations,
        "full_best_k": full_best_k,
        "full_cluster_feature_associations": full_cluster_association,
        "k_evaluations": evaluations,
        "best_k": best_k,
        "final_silhouette": float(final_silhouette),
        "final_davies_bouldin": float(final_db),
        "final_stability": final_stability,
        "profiles": profile_definitions,
        "cluster_feature_associations": cluster_association,
        "risk_cluster_chi_square": {
            "chi_square": float(chi2),
            "degrees_of_freedom": int((risk_table.shape[0] - 1) * (risk_table.shape[1] - 1)),
            "p_value": float(risk_p),
            "cramers_v": risk_cramers_v,
            "table": risk_table.to_dict(),
        },
        "individual_factor_associations": individual_association,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
