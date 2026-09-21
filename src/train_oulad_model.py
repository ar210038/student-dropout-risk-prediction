"""Train and evaluate leakage-safe OULAD withdrawal-risk candidates."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from oulad_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    PREDICTION_CUTOFF,
    TARGET,
    build_modeling_dataset,
    grouped_holdout,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "oulad" / "raw"
MODEL_DIR = ROOT / "models" / "oulad"
REPORT_DIR = ROOT / "reports" / "oulad"
FIGURE_DIR = REPORT_DIR / "figures"
THRESHOLDS = np.arange(0.20, 0.61, 0.05)


def preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ]
    )


def candidates(scale_pos_weight: float) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", preprocessor(True)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", preprocessor(False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=350,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocess", preprocessor(False)),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=450,
                        max_depth=4,
                        learning_rate=0.04,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=5,
                        reg_lambda=2.0,
                        # Threshold tuning handles the intervention tradeoff; keeping
                        # the natural prevalence gives more interpretable probabilities.
                        scale_pos_weight=1.0,
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def metrics(y: pd.Series, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
        "average_precision": average_precision_score(y, probabilities),
        "brier_score": brier_score_loss(y, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_positive_rate": float(predictions.mean()),
    }


def grouped_oof(model: Pipeline, train: pd.DataFrame) -> np.ndarray:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = np.zeros(len(train))
    X, y, groups = train[FEATURES], train[TARGET], train["id_student"]
    for fit_idx, valid_idx in splitter.split(X, y, groups):
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        probabilities[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return probabilities


def choose_threshold(y: pd.Series, probabilities: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for threshold in THRESHOLDS:
        row = metrics(y, probabilities, float(threshold))
        row["threshold"] = float(round(threshold, 2))
        rows.append(row)
    eligible = [row for row in rows if row["recall"] >= 0.70]
    pool = eligible or rows
    chosen = max(pool, key=lambda row: (row["f1"], row["precision"]))
    return chosen["threshold"], rows


def top_features(model: Pipeline, limit: int = 15) -> list[dict]:
    names = model.named_steps["preprocess"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    values = (
        classifier.coef_[0]
        if hasattr(classifier, "coef_")
        else classifier.feature_importances_
    )
    order = np.argsort(np.abs(values))[::-1][:limit]
    return [
        {"feature": str(names[index]), "association": float(values[index])}
        for index in order
    ]


def save_figures(y: pd.Series, probabilities: np.ndarray, threshold: float) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fraction_positive, mean_predicted = calibration_curve(
        y, probabilities, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_predicted, fraction_positive, marker="o", label="Selected model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed withdrawal rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, probabilities)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "precision_recall_curve.png", dpi=160)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, cohort_stats = build_modeling_dataset(RAW)
    train, test = grouped_holdout(dataset)
    overlap = set(train["id_student"]) & set(test["id_student"])
    if overlap:
        raise RuntimeError("Student leakage between train and test")

    cv_results, oof_by_model = {}, {}
    class_ratio = float((train[TARGET] == 0).sum() / (train[TARGET] == 1).sum())
    model_candidates = candidates(class_ratio)
    for name, model in model_candidates.items():
        oof = grouped_oof(model, train)
        threshold, threshold_rows = choose_threshold(train[TARGET], oof)
        cv_results[name] = {
            "metrics_at_selected_threshold": metrics(train[TARGET], oof, threshold),
            "selected_threshold": threshold,
            "threshold_analysis": threshold_rows,
        }
        oof_by_model[name] = oof

    selected_name = max(
        cv_results,
        key=lambda name: (
            cv_results[name]["metrics_at_selected_threshold"]["average_precision"],
            cv_results[name]["metrics_at_selected_threshold"]["f1"],
        ),
    )
    selected_threshold = cv_results[selected_name]["selected_threshold"]
    selected_model = candidates(class_ratio)[selected_name]
    selected_model.fit(train[FEATURES], train[TARGET])
    test_probabilities = selected_model.predict_proba(test[FEATURES])[:, 1]
    test_metrics = metrics(test[TARGET], test_probabilities, selected_threshold)
    importance = top_features(selected_model)
    save_figures(test[TARGET], test_probabilities, selected_threshold)

    result = {
        "dataset": cohort_stats,
        "prediction_cutoff_days": PREDICTION_CUTOFF,
        "target": "final_result == 'Withdrawn'",
        "features": FEATURES,
        "train_records": len(train),
        "test_records": len(test),
        "train_unique_students": train["id_student"].nunique(),
        "test_unique_students": test["id_student"].nunique(),
        "student_overlap": len(overlap),
        "class_distribution": {
            "withdrawn": int(dataset[TARGET].sum()),
            "non_withdrawn": int((dataset[TARGET] == 0).sum()),
            "withdrawn_percent": float(100 * dataset[TARGET].mean()),
        },
        "cv_results": cv_results,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "test_metrics": test_metrics,
        "top_features": importance,
        "xgboost_used": True,
        "selection_rule": "Highest grouped out-of-fold average precision; threshold maximizes F1 subject to recall >= 0.70.",
    }
    # A deliberately strict migration gate prevents a weak experiment from becoming
    # a deployable artifact merely because it ranks risks better than chance.
    meets_candidate_gate = (
        test_metrics["precision"] >= 0.35
        and test_metrics["recall"] >= 0.70
        and test_metrics["average_precision"] >= 0.40
        and test_metrics["brier_score"] <= 0.18
    )
    metadata = {
        "candidate_only": True,
        "model": selected_name,
        "threshold": selected_threshold,
        "prediction_cutoff_days": PREDICTION_CUTOFF,
        "target": "course withdrawal after day 30",
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "test_metrics": test_metrics,
        "candidate_gate_passed": meets_candidate_gate,
    }
    result["candidate_gate_passed"] = meets_candidate_gate
    with (REPORT_DIR / "training_results.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    if meets_candidate_gate:
        joblib.dump(selected_model, MODEL_DIR / "oulad_dropout_model.joblib")
        with (MODEL_DIR / "oulad_model_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
    else:
        (MODEL_DIR / "oulad_dropout_model.joblib").unlink(missing_ok=True)
        (MODEL_DIR / "oulad_model_metadata.json").unlink(missing_ok=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
