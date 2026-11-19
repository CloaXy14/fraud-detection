"""Suspicious credit-card transaction detection.

Fits Random Forest or Isolation Forest on chronological partitions,
evaluates a labeled holdout, and exports ranked review queues from saved models.
See README.md and plan.md for the experiment design.

Subcommands:
    train     Fit a detector on earlier transactions and freeze validation cutoffs.
    evaluate  Score the held-out test partition once with the frozen run.
    score     Score a schema-compatible CSV and export a ranked review queue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
RAW_INPUT_FIELDS = ["Time"] + FEATURES  # the 30 raw input fields
LABEL = "Class"
TIME = "Time"
ROW = "row_number"

SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
PARTITIONS = ("train", "validation", "test")

# Review budgets as fractions of validation rows. Labels are used as JSON keys.
BUDGETS = {"0.1%": "0.001", "0.5%": "0.005", "1.0%": "0.01"}
MAIN_BUDGET = "0.5%"

IFOREST_PARAMS = {
    "n_estimators": 100,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": 42,
}

BASELINE_MODEL = "isolation_forest"
METHODS = (BASELINE_MODEL, "amount_only")
RANDOM_FOREST_PARAMS = {"n_estimators": 300, "n_jobs": -1, "random_state": 42}
IF_COMBINED_PARAMS = {**IFOREST_PARAMS, "n_estimators": 300, "max_samples": 2048, "max_features": 0.75}
SCORE_COLUMNS = {BASELINE_MODEL: "anomaly_score", "random_forest": "fraud_score", "if_combined": "anomaly_score"}
METHOD_TITLES = {
    BASELINE_MODEL: "Isolation Forest", "amount_only": "Amount only",
    "random_forest": "Random Forest", "if_combined": "Tuned Isolation Forest",
}

PUBLISHER_FIGURES = {"rows": 284807, "frauds": 492}
DATASET_SOURCE = {
    "publisher_listing": "https://www.kaggle.com/mlg-ulb/creditcardfraud/",
    "publisher": "Machine Learning Group, Universite Libre de Bruxelles, and Worldline",
    "download_url": "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
    "download_note": (
        "The Kaggle listing requires a login. The file was retrieved from the public "
        "copy that TensorFlow's imbalanced-data tutorial links to, then checked against "
        "the publisher's row and fraud counts."
    ),
    "license": "Database Contents License (DbCL) v1.0, as reported for the Kaggle listing",
}

MODEL_FORMAT_VERSION = 1
PROMOTED_MODEL_FORMAT_VERSION = 2


class DataValidationError(ValueError):
    """Raised when an input file fails schema or value checks."""


# --------------------------------------------------------------------------
# Input validation and audit
# --------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_rows(row_numbers, limit: int = 5) -> str:
    rows = [str(int(r)) for r in row_numbers[:limit]]
    more = len(row_numbers) - limit
    return ", ".join(rows) + (f" (and {more} more)" if more > 0 else "")


def read_header(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise DataValidationError(f"{path}: input file not found")
    with open(path, newline="", encoding="utf-8-sig") as handle:
        try:
            header = next(csv.reader(handle))
        except StopIteration:
            raise DataValidationError(f"{path}: file is empty (no header row)") from None
    header = [name.strip() for name in header]
    if not header or all(name == "" for name in header):
        raise DataValidationError(f"{path}: file is empty (no header row)")
    seen, duplicates = set(), []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise DataValidationError(f"{path}: duplicate header name(s): {', '.join(duplicates)}")
    return header


def load_transactions(
    path: str,
    require_label: bool,
    require_time: bool,
    features=FEATURES,
) -> pd.DataFrame:
    """Read and validate a transaction CSV.

    Returns a DataFrame with ``row_number`` (1 = first record after the header),
    numeric float64 feature columns, ``Time`` when present, and an integer
    ``Class`` when ``require_label`` is set. Columns are matched by name, so their
    order in the file does not matter. Extra columns are ignored.
    """
    header = read_header(path)
    required = list(features)
    if require_time:
        required = [TIME] + required
    if require_label:
        required = required + [LABEL]
    missing = [name for name in required if name not in header]
    if missing:
        raise DataValidationError(f"{path}: missing required column(s): {', '.join(missing)}")

    try:
        raw = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=False,
            skipinitialspace=True,
            encoding="utf-8-sig",
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataValidationError(f"{path}: malformed CSV: {exc}") from None
    raw.columns = [str(c).strip() for c in raw.columns]
    if len(raw) == 0:
        raise DataValidationError(f"{path}: no transaction rows after the header")

    row_numbers = np.arange(1, len(raw) + 1, dtype=np.int64)
    numeric_columns = list(features)
    if TIME in raw.columns and TIME not in numeric_columns:
        numeric_columns = [TIME] + numeric_columns
    use_label = require_label
    if use_label and LABEL not in numeric_columns:
        numeric_columns = numeric_columns + [LABEL]

    problems = []
    out = pd.DataFrame({ROW: row_numbers})
    for column in numeric_columns:
        text = raw[column]
        is_missing = text.isna() | (text.str.strip() == "")
        values = pd.to_numeric(text.where(~is_missing), errors="coerce")
        nonnumeric = values.isna() & ~is_missing
        infinite = np.isinf(values.to_numpy(dtype=float, na_value=np.nan))
        for mask, kind in ((is_missing, "missing"), (nonnumeric, "nonnumeric or NaN"), (infinite, "infinite")):
            mask = np.asarray(mask)
            if mask.any():
                problems.append(f"column '{column}': {int(mask.sum())} {kind} value(s) at row(s) {_format_rows(row_numbers[mask])}")
        values = values.to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(values)
        if column in (TIME, "Amount"):
            negative = valid & (values < 0)
            if negative.any():
                problems.append(f"column '{column}': {int(negative.sum())} negative value(s) at row(s) {_format_rows(row_numbers[negative])}")
        if column == LABEL:
            bad = valid & ~np.isin(values, (0.0, 1.0))
            if bad.any():
                problems.append(f"column '{column}': {int(bad.sum())} value(s) other than 0 or 1 at row(s) {_format_rows(row_numbers[bad])}")
        out[column] = values
    if problems:
        raise DataValidationError(f"{path}: invalid input\n  " + "\n  ".join(problems))
    if use_label:
        out[LABEL] = out[LABEL].astype(np.int64)
    return out


def audit_dataset(df: pd.DataFrame) -> dict:
    labels = df[LABEL].to_numpy()
    n = int(len(df))
    frauds = int(labels.sum())
    full_columns = RAW_INPUT_FIELDS + [LABEL]
    exact_dupes = int(df.duplicated(subset=full_columns, keep="first").sum())
    rows_in_exact_groups = int(df.duplicated(subset=full_columns, keep=False).sum())
    by_fields = df.groupby(RAW_INPUT_FIELDS, sort=False)[LABEL].nunique()
    conflicting_groups = int((by_fields > 1).sum())
    conflicting_rows = 0
    if conflicting_groups:
        conflict_mask = df.groupby(RAW_INPUT_FIELDS, sort=False)[LABEL].transform("nunique") > 1
        conflicting_rows = int(conflict_mask.sum())
    return {
        "rows": n,
        "columns_checked": full_columns,
        "label_counts": {"0": n - frauds, "1": frauds},
        "fraud_prevalence": frauds / n if n else None,
        "missing_values": {c: int(df[c].isna().sum()) for c in full_columns},
        "time_range_seconds": [float(df[TIME].min()), float(df[TIME].max())],
        "amount_range": [float(df["Amount"].min()), float(df["Amount"].max())],
        "exact_duplicate_records_beyond_first": exact_dupes,
        "records_in_exact_duplicate_groups": rows_in_exact_groups,
        "raw_field_groups_with_conflicting_labels": conflicting_groups,
        "records_in_conflicting_label_groups": conflicting_rows,
        "duplicate_handling": "Records kept as published; equal Time values stay in one partition.",
        "publisher_comparison": {
            "expected_rows": PUBLISHER_FIGURES["rows"],
            "expected_frauds": PUBLISHER_FIGURES["frauds"],
            "rows_match": n == PUBLISHER_FIGURES["rows"],
            "frauds_match": frauds == PUBLISHER_FIGURES["frauds"],
        },
    }


# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------


def _move_past_group(times: np.ndarray, boundary: int) -> int:
    n = len(times)
    while 0 < boundary < n and times[boundary] == times[boundary - 1]:
        boundary += 1
    return boundary


def _row_hash(row_numbers) -> str:
    return hashlib.sha256(np.asarray(row_numbers, dtype="<i8").tobytes()).hexdigest()


def chronological_split(df: pd.DataFrame, fractions=SPLIT_FRACTIONS):
    """Split by Time without cutting a timestamp group.

    Returns ``(parts, manifest)`` where ``parts`` maps partition name to a
    DataFrame sorted by Time then row_number.
    """
    ordered = df.sort_values([TIME, ROW], kind="mergesort").reset_index(drop=True)
    times = ordered[TIME].to_numpy()
    n = len(ordered)
    frac0 = Fraction(str(fractions[0]))
    frac01 = frac0 + Fraction(str(fractions[1]))
    first = _move_past_group(times, math.floor(n * frac0))
    second = _move_past_group(times, max(first, math.floor(n * frac01)))
    bounds = {"train": (0, first), "validation": (first, second), "test": (second, n)}
    parts = {name: ordered.iloc[a:b].reset_index(drop=True) for name, (a, b) in bounds.items()}
    for name, part in parts.items():
        if len(part) == 0:
            raise DataValidationError(f"chronological split produced an empty {name} partition ({n} rows total)")
    if not parts["train"][TIME].max() < parts["validation"][TIME].min():
        raise AssertionError("train and validation timestamps overlap")
    if not parts["validation"][TIME].max() < parts["test"][TIME].min():
        raise AssertionError("validation and test timestamps overlap")

    manifest = {
        "method": "sort by Time then row_number; boundaries at floor(0.6n) and floor(0.8n), moved forward past the timestamp group they cut",
        "requested_fractions": list(fractions),
        "total_rows": n,
        "boundary_indices": [first, second],
        "partitions": {},
    }
    for name, part in parts.items():
        manifest["partitions"][name] = {
            "rows": int(len(part)),
            "fraction": len(part) / n,
            "time_range_seconds": [float(part[TIME].min()), float(part[TIME].max())],
            "row_number_sha256": _row_hash(part[ROW].to_numpy()),
        }
    return parts, manifest


def class_counts(labels) -> dict:
    labels = np.asarray(labels)
    positives = int(labels.sum())
    return {"0": int(len(labels) - positives), "1": positives}


# --------------------------------------------------------------------------
# Scores, cutoffs and metrics
# --------------------------------------------------------------------------


def feature_matrix(df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    return df.loc[:, list(features)]


def isolation_scores(estimator, df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    """Larger means more unusual: anomaly_score = -score_samples(X)."""
    return -estimator.score_samples(feature_matrix(df, features))


def model_scores(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    estimator, features = bundle["estimator"], bundle["features"]
    if bundle.get("model_kind") == "random_forest":
        scores = estimator.predict_proba(feature_matrix(df, features))[:, 1]
    else:
        scores = isolation_scores(estimator, df, features)
    if not np.all(np.isfinite(scores)):
        raise DataValidationError("model scores must be finite")
    return scores


def amount_scores(df: pd.DataFrame) -> np.ndarray:
    return df["Amount"].to_numpy(dtype=float)


def budget_size(rate: str, n: int) -> int:
    """K = floor(rate * n), computed exactly from the decimal rate string."""
    return math.floor(Fraction(rate) * n)


def select_cutoff(scores, rate: str) -> dict:
    """Pick the cutoff that flags the most whole tied-score groups within K rows."""
    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    n = len(scores)
    k = budget_size(rate, n)
    result = {"rate": rate, "rows": n, "K": k, "cutoff": None, "flagged": 0, "active": False}
    if k == 0 or n == 0:
        result["reason"] = "K = 0"
        return result
    values, counts = np.unique(scores, return_counts=True)
    values, counts = values[::-1], counts[::-1]
    cumulative = np.cumsum(counts)
    fitting = int(np.searchsorted(cumulative, k, side="right"))
    if fitting == 0:
        result["reason"] = f"highest tied group has {int(counts[0])} rows, more than K = {k}"
        return result
    result.update(cutoff=float(values[fitting - 1]), flagged=int(cumulative[fitting - 1]), active=True)
    return result


def apply_cutoff(scores, cutoff) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if cutoff is None:
        return np.zeros(len(scores), dtype=np.int64)
    return (scores >= cutoff).astype(np.int64)


def ranking_order(scores, row_numbers) -> np.ndarray:
    """Indices sorted by score descending, then row_number ascending."""
    return np.lexsort((np.asarray(row_numbers), -np.asarray(scores, dtype=float)))


def _ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def confusion_metrics(labels, flags) -> dict:
    labels = np.asarray(labels).astype(bool)
    flags = np.asarray(flags).astype(bool)
    tp = int(np.sum(labels & flags))
    fp = int(np.sum(~labels & flags))
    fn = int(np.sum(labels & ~flags))
    tn = int(np.sum(~labels & ~flags))
    n = int(len(labels))
    positives = int(labels.sum())
    if tp + fp + fn + tn != n or tp + fn != positives:
        raise AssertionError("confusion counts do not reconcile with the input")
    alerts = tp + fp
    alert_rate = _ratio(alerts, n)
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "total": n,
        "positive_support": positives,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "alerts": alerts,
        "alert_rate": alert_rate,
        "alerts_per_100k": None if alert_rate is None else alert_rate * 100_000,
    }


def top_k_metrics(labels, scores, row_numbers, rate: str) -> dict:
    labels = np.asarray(labels).astype(bool)
    k = budget_size(rate, len(labels))
    order = ranking_order(scores, row_numbers)[:k]
    tp = int(labels[order].sum())
    positives = int(labels.sum())
    return {
        "rate": rate, "K": k, "TP_in_top_K": tp,
        "precision_at_K": _ratio(tp, k),
        "recall_at_K": _ratio(tp, positives),
    }


def average_precision(labels, scores) -> dict:
    from sklearn.metrics import average_precision_score

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return {"AP": None, "available": False, "reason": "partition has a single class"}
    return {"AP": float(average_precision_score(labels, scores)), "available": True}


def fmt_rate(value) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def fmt_number(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------------
# Helpers for run metadata and artifacts
# --------------------------------------------------------------------------


def environment_versions() -> dict:
    import joblib
    import matplotlib
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def quantiles(values) -> dict:
    points = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 0.995, 0.999, 1.0]
    values = np.asarray(values, dtype=float)
    return {str(p): float(np.quantile(values, p)) for p in points}


def plot_precision_recall(curves: list, prevalence, title: str, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    for label, labels, scores, ap in curves:
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.step(recall, precision, where="post", label=f"{label} (AP = {ap:.3f})")
    ax.axhline(prevalence, color="grey", linestyle="--", label=f"No-skill: fraud prevalence = {prevalence:.4f}")
    ax.set_xlabel("Recall (share of recorded frauds caught)")
    ax.set_ylabel("Precision (share of flagged rows that are fraud)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_development_overview(parts: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = pd.concat([parts["train"], parts["validation"]], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=110)
    bins = np.logspace(-2, np.log10(max(dev["Amount"].max(), 1.0) + 1), 60)
    for cls, color, name in ((0, "tab:blue", "Legitimate"), (1, "tab:red", "Fraud")):
        amounts = dev.loc[dev[LABEL] == cls, "Amount"] + 0.01
        if len(amounts) > 0:
            axes[0].hist(amounts, bins=bins, density=True, alpha=0.5, color=color, label=f"{name} (n={len(amounts)})")
        else:
            axes[0].plot([], [], color=color, alpha=0.5, label=f"{name} (n=0)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Amount + 0.01 (log scale)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Amount distribution, train + validation")
    axes[0].legend()
    hours = (dev[TIME] // 3600).astype(int)
    per_hour = dev.groupby(hours)[LABEL].sum()
    axes[1].bar(per_hour.index, per_hour.values, color="tab:red")
    boundary = parts["validation"][TIME].min() / 3600
    axes[1].axvline(boundary, color="black", linestyle="--", label="Validation starts")
    axes[1].set_xlabel("Hours since first transaction")
    axes[1].set_ylabel("Labeled frauds")
    axes[1].set_title("Labeled frauds per hour, train + validation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def development_inspection(parts: dict) -> dict:
    summary = {}
    for name in ("train", "validation"):
        part = parts[name]
        entry = {"class_counts": class_counts(part[LABEL])}
        entry["amount_quantiles_by_class"] = {
            str(cls): quantiles(part.loc[part[LABEL] == cls, "Amount"]) if (part[LABEL] == cls).any() else None
            for cls in (0, 1)
        }
        hours = (part[TIME] // 3600).astype(int)
        entry["frauds_per_hour"] = {str(h): int(v) for h, v in part.groupby(hours)[LABEL].sum().items()}
        summary[name] = entry
    return summary


def _publish_dir(staging: str, destination: str) -> None:
    if os.path.exists(destination):
        raise FileExistsError(f"{destination} appeared while training; staged results kept at {staging}")
    os.rename(staging, destination)


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


def evaluate_methods(labels, row_numbers, method_scores: dict, cutoffs: dict) -> dict:
    """Cutoff metrics at frozen cutoffs plus fixed-size ranking diagnostics."""
    result = {}
    for method, scores in method_scores.items():
        entry = {"average_precision": average_precision(labels, scores), "budgets": {}}
        for budget, rate in BUDGETS.items():
            cutoff = cutoffs[method][budget]["cutoff"]
            flags = apply_cutoff(scores, cutoff)
            entry["budgets"][budget] = {
                "frozen_cutoff": cutoff,
                "at_cutoff": confusion_metrics(labels, flags),
                "fixed_size_ranking": top_k_metrics(labels, scores, row_numbers, rate),
            }
        result[method] = entry
    return result


def train(input_path: str, run_dir: str, model_kind: str = BASELINE_MODEL) -> dict:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    import joblib

    started = time.perf_counter()
    run_dir = os.path.abspath(run_dir)
    if os.path.exists(run_dir):
        raise FileExistsError(f"{run_dir} already exists; choose a new --run-dir so earlier runs are preserved")

    if model_kind not in SCORE_COLUMNS:
        raise ValueError(f"unknown model kind {model_kind!r}")
    baseline = model_kind == BASELINE_MODEL
    supervised = model_kind == "random_forest"
    tuned = model_kind == "if_combined"
    features = FEATURES[:-1] if tuned else FEATURES
    labeled = not tuned or LABEL in read_header(input_path)
    df = load_transactions(input_path, require_label=labeled, require_time=True, features=features)
    data_hash = sha256_file(input_path)
    audit = audit_dataset(df) if baseline else {"rows": len(df), "columns_checked": list(df.columns.drop(ROW))}
    parts, manifest = chronological_split(df)
    if labeled:
        audit["label_counts"] = class_counts(df[LABEL])
        for name in ("train", "validation"):
            manifest["partitions"][name]["class_counts"] = class_counts(parts[name][LABEL])

    fit = parts["train"]
    fit_started = time.perf_counter()
    if supervised:
        y = fit[LABEL]
        if y.nunique() != 2:
            raise DataValidationError("Random Forest training rows must contain both Class 0 and Class 1")
        settings = RANDOM_FOREST_PARAMS
        estimator = RandomForestClassifier(**settings).fit(feature_matrix(fit, features), y)
        # Serial inference keeps probability summation stable at exact cutoffs.
        estimator.set_params(n_jobs=1)
    else:
        settings = IF_COMBINED_PARAMS if tuned else IFOREST_PARAMS
        estimator = IsolationForest(**settings).fit(feature_matrix(fit, features))
    fit_seconds = time.perf_counter() - fit_started
    fitted_features = estimator.feature_names_in_.tolist()
    if fitted_features != features:
        raise AssertionError(f"fitted features {fitted_features} differ from {features}")

    model_bundle = {"estimator": estimator, "features": fitted_features}
    if not baseline:
        model_bundle["model_kind"] = model_kind
    validation = parts["validation"]
    val_labels = validation[LABEL].to_numpy() if labeled else None
    val_scores = {model_kind: model_scores(model_bundle, validation)}
    if baseline:
        val_scores["amount_only"] = amount_scores(validation)
    cutoffs = {m: {b: select_cutoff(s, r) for b, r in BUDGETS.items()} for m, s in val_scores.items()}
    validation_metrics = evaluate_methods(val_labels, validation[ROW], val_scores, cutoffs) if labeled else None
    val_single_class = len(np.unique(val_labels)) < 2 if labeled else None

    versions = environment_versions()
    run_id = f"{os.path.basename(run_dir)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    dataset = {
        "path": os.path.abspath(input_path),
        "sha256": data_hash,
        "size_bytes": os.path.getsize(input_path),
        "retrieved_utc": datetime.fromtimestamp(os.path.getmtime(input_path), timezone.utc).isoformat(timespec="seconds"),
        **(DATASET_SOURCE if baseline else {}),
    }
    selection = {
        "main_budget": MAIN_BUDGET,
        "rule": (
            "For each method and budget, K = floor(rate * validation rows). Validation scores are sorted "
            "descending and grouped by exact ties; the cutoff is the lowest score of the largest set of whole "
            "groups whose size is at most K. Rows with score >= cutoff are flagged. No active cutoff when K = 0 "
            "or the top tied group exceeds K."
        ),
        "model_choice": (
            "First-release configuration only: the plan's fixed Isolation Forest settings and seed, all training "
            "rows, 29 features. No variants were compared, so no validation-driven model selection occurred."
        ) if baseline else f"Recorded benchmark recipe for {model_kind}.",
        "review_flag_source": f"{model_kind} cutoff at the {MAIN_BUDGET} validation budget",
    }
    score_column = SCORE_COLUMNS[model_kind]
    model_bundle.update({
        "format_version": MODEL_FORMAT_VERSION if baseline else PROMOTED_MODEL_FORMAT_VERSION,
        "run_id": run_id,
        "score_definition": (
            "fraud_score = RandomForestClassifier.predict_proba(X)[:, 1]; class-1 ranking score"
            if supervised else "anomaly_score = -IsolationForest.score_samples(X); larger is more unusual"
        ),
        "flag_rule": f"review_flag = 1 when {score_column} >= cutoff",
        "main_budget": MAIN_BUDGET,
        "cutoffs": cutoffs,
        "settings": dict(settings),
        "versions": versions,
        "dataset": {"sha256": data_hash, "rows": audit["rows"]},
        "split": {name: manifest["partitions"][name]["row_number_sha256"] for name in PARTITIONS},
    })

    os.makedirs(os.path.dirname(run_dir) or ".", exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{os.path.basename(run_dir)}.staging-", dir=os.path.dirname(run_dir) or ".")
    try:
        joblib.dump(model_bundle, os.path.join(staging, "model.joblib"))
        if baseline:
            plot_development_overview(parts, os.path.join(staging, "development_overview.png"))
        if labeled and not val_single_class:
            plot_precision_recall(
                [(METHOD_TITLES[m], val_labels, s, validation_metrics[m]["average_precision"]["AP"])
                 for m, s in val_scores.items()],
                float(val_labels.mean()),
                "Validation precision-recall",
                os.path.join(staging, "validation_precision_recall.png"),
            )
        run_record = {
            "run_id": run_id,
            "created_utc": _utc_now(),
            "command": " ".join([os.path.basename(sys.executable)] + sys.argv) if sys.argv else None,
            "dataset": dataset,
            "audit": audit,
            "split": manifest,
            "development_inspection": development_inspection(parts) if baseline else None,
            "model": {
                "type": f"sklearn.ensemble.{type(estimator).__name__}",
                "parameters": dict(settings),
                "features": fitted_features,
                "excluded_from_features": [TIME, LABEL, ROW],
                "fit_seconds": fit_seconds,
            },
            "versions": versions,
            "cutoff_selection": selection,
            "validation": {
                "rows": int(len(validation)),
                "class_counts": class_counts(val_labels) if labeled else None,
                "fraud_prevalence": float(val_labels.mean()) if labeled else None,
                "single_class": val_single_class,
                "cutoffs": cutoffs,
                "metrics": validation_metrics,
                f"{model_kind}_score_quantiles": quantiles(val_scores[model_kind]),
            },
            "duration_seconds": None,
        }
        if not baseline:
            run_record["model"].update(
                kind=model_kind, bundle_version=PROMOTED_MODEL_FORMAT_VERSION,
                score_definition=model_bundle["score_definition"],
            )
        if supervised:
            run_record["model"]["inference_n_jobs"] = estimator.n_jobs
        run_record["duration_seconds"] = time.perf_counter() - started
        write_json(os.path.join(staging, "run.json"), run_record)
        _publish_dir(staging, run_dir)
    except BaseException:
        if os.path.isdir(staging) and not os.path.exists(run_dir):
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return run_record


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def load_model(path: str) -> dict:
    import joblib

    bundle = joblib.load(path)  # trusted local files only; joblib can execute code
    if isinstance(bundle, dict):
        version = bundle.get("format_version")
        if version == MODEL_FORMAT_VERSION and bundle.get("model_kind", BASELINE_MODEL) == BASELINE_MODEL:
            return bundle
        if version == PROMOTED_MODEL_FORMAT_VERSION and bundle.get("model_kind") in ("random_forest", "if_combined"):
            return bundle
    raise ValueError(f"{path}: not a supported model file written by this tool")


def _row_summary(part: pd.DataFrame, scores: np.ndarray, mask: np.ndarray, ascending: bool, limit: int = 10) -> dict:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"count": 0, "amount_range": None, "amount_median": None, "time_window_seconds": None, "examples": []}
    order = idx[np.lexsort((part[ROW].to_numpy()[idx], scores[idx] if ascending else -scores[idx]))]
    amounts = part["Amount"].to_numpy()[idx]
    times = part[TIME].to_numpy()[idx]
    examples = [
        {"row_number": int(part[ROW].iat[i]), "anomaly_score": float(scores[i]), "Amount": float(part["Amount"].iat[i]), "Time": float(part[TIME].iat[i])}
        for i in order[:limit]
    ]
    return {
        "count": int(len(idx)),
        "amount_range": [float(amounts.min()), float(amounts.max())],
        "amount_median": float(np.median(amounts)),
        "time_window_seconds": [float(times.min()), float(times.max())],
        "examples": examples,
    }


def error_analysis(test: pd.DataFrame, scores: dict, cutoffs: dict) -> dict:
    labels = test[LABEL].to_numpy().astype(bool)
    flags = {m: apply_cutoff(scores[m], cutoffs[m][MAIN_BUDGET]["cutoff"]).astype(bool) for m in METHODS}
    if_scores = scores["isolation_forest"]
    amounts = test["Amount"].to_numpy()
    q25, q75 = np.quantile(amounts, [0.25, 0.75])
    moderate = (amounts >= q25) & (amounts <= q75)
    caught_if, caught_amount = flags["isolation_forest"] & labels, flags["amount_only"] & labels
    return {
        "budget": MAIN_BUDGET,
        "isolation_forest_false_positives_highest_scores": _row_summary(test, if_scores, flags["isolation_forest"] & ~labels, ascending=False),
        "isolation_forest_false_negatives_lowest_scores": _row_summary(test, if_scores, ~flags["isolation_forest"] & labels, ascending=True),
        "method_disagreement": {
            "frauds_total": int(labels.sum()),
            "caught_by_both": int(np.sum(caught_if & caught_amount)),
            "isolation_forest_only": int(np.sum(caught_if & ~caught_amount)),
            "amount_only_only": int(np.sum(caught_amount & ~caught_if)),
            "caught_by_neither": int(np.sum(labels & ~flags["isolation_forest"] & ~flags["amount_only"])),
            "moderate_amount_range": [float(q25), float(q75)],
            "moderate_amount_frauds": int(np.sum(labels & moderate)),
            "moderate_amount_frauds_isolation_forest_only": int(np.sum(caught_if & ~caught_amount & moderate)),
            "legitimate_flagged_by_both": int(np.sum(~labels & flags["isolation_forest"] & flags["amount_only"])),
        },
    }


def evaluate(input_path: str, run_dir: str) -> dict:
    started = time.perf_counter()
    run_dir = os.path.abspath(run_dir)
    model_path = os.path.join(run_dir, "model.joblib")
    run_path = os.path.join(run_dir, "run.json")
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(model_path) or not os.path.isfile(run_path):
        raise FileNotFoundError(f"{run_dir} has no model.joblib/run.json; run train first")
    if os.path.exists(metrics_path):
        raise FileExistsError(f"{metrics_path} already exists; the test partition is evaluated once per run")
    bundle = load_model(model_path)
    with open(run_path, encoding="utf-8") as handle:
        run_record = json.load(handle)
    if run_record.get("run_id") != bundle["run_id"]:
        raise DataValidationError(f"{run_dir}: run.json run_id does not match model.joblib")
    if run_record.get("dataset", {}).get("sha256") != bundle["dataset"]["sha256"]:
        raise DataValidationError(f"{run_dir}: run.json dataset SHA-256 does not match model.joblib")

    data_hash = sha256_file(input_path)
    if data_hash != bundle["dataset"]["sha256"]:
        raise DataValidationError(
            f"{input_path}: SHA-256 {data_hash} does not match the training dataset {bundle['dataset']['sha256']}"
        )
    df = load_transactions(input_path, require_label=True, require_time=True, features=bundle["features"])
    parts, manifest = chronological_split(df)
    for name in PARTITIONS:
        if manifest["partitions"][name]["row_number_sha256"] != bundle["split"][name]:
            raise DataValidationError(f"{name} partition differs from the saved split")

    test = parts["test"]
    labels = test[LABEL].to_numpy()
    rows = test[ROW].to_numpy()
    model_kind = bundle.get("model_kind", BASELINE_MODEL)
    scores = {model_kind: model_scores(bundle, test)}
    if model_kind == BASELINE_MODEL:
        scores["amount_only"] = amount_scores(test)
    cutoffs = bundle["cutoffs"]
    test_metrics = evaluate_methods(labels, rows, scores, cutoffs)
    single_class = len(np.unique(labels)) < 2
    prevalence = float(labels.mean())

    partition_scores = {
        "train": model_scores(bundle, parts["train"]),
        "validation": model_scores(bundle, parts["validation"]),
        "test": scores[model_kind],
    }
    val_metrics = run_record["validation"]["metrics"]
    alert_rates = {
        m: {
            b: {
                "validation": val_metrics[m]["budgets"][b]["at_cutoff"]["alert_rate"],
                "test": test_metrics[m]["budgets"][b]["at_cutoff"]["alert_rate"],
            }
            for b in BUDGETS
        }
        for m in scores
    }

    metrics = {
        "run_id": bundle["run_id"],
        "created_utc": _utc_now(),
        "dataset_sha256": data_hash,
        "score_definition": bundle["score_definition"],
        "flag_rule": bundle["flag_rule"],
        "model_settings": bundle["settings"],
        "test": {
            "time_range_seconds": manifest["partitions"]["test"]["time_range_seconds"],
            "rows": int(len(test)),
            "class_counts": class_counts(labels),
            "fraud_prevalence": prevalence,
            "no_skill_reference": {
                "precision_equals_prevalence": prevalence,
                "empty_queue_frauds_caught": 0,
            },
            "single_class": single_class,
            "ranking_comparison_available": not single_class,
        },
        "methods": test_metrics,
        f"{model_kind}_score_quantiles": {k: quantiles(v) for k, v in partition_scores.items()},
        "alert_rates_validation_vs_test": alert_rates,
        "duration_seconds": None,
    }
    if model_kind == BASELINE_MODEL:
        metrics["error_analysis"] = error_analysis(test, scores, cutoffs)

    staging = tempfile.mkdtemp(prefix=".evaluate-staging-", dir=run_dir)
    try:
        if not single_class:
            plot_precision_recall(
                [(METHOD_TITLES[m], labels, s, test_metrics[m]["average_precision"]["AP"])
                 for m, s in scores.items()],
                prevalence,
                "Held-out test precision-recall (latest 20% of transactions)",
                os.path.join(staging, "precision_recall.png"),
            )
        metrics["duration_seconds"] = time.perf_counter() - started
        write_json(os.path.join(staging, "metrics.json"), metrics)
        for name in os.listdir(staging):
            os.replace(os.path.join(staging, name), os.path.join(run_dir, name))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return metrics


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def _same_path(a: str, b: str) -> bool:
    if os.path.realpath(a) == os.path.realpath(b):
        return True
    return os.path.exists(a) and os.path.exists(b) and os.path.samefile(a, b)


def score_frame(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    model_kind = bundle.get("model_kind", BASELINE_MODEL)
    scores = model_scores(bundle, df)
    cutoff = bundle["cutoffs"][model_kind][bundle["main_budget"]]["cutoff"]
    out = pd.DataFrame({
        ROW: df[ROW].to_numpy(),
        SCORE_COLUMNS[model_kind]: scores,
        "review_flag": apply_cutoff(scores, cutoff),
    })
    return out.iloc[ranking_order(scores, out[ROW].to_numpy())].reset_index(drop=True)


def score(input_path: str, model_path: str, output_path: str) -> dict:
    if os.path.isdir(output_path):
        raise ValueError(f"--output {output_path} is a directory; provide a destination CSV path")
    for other, name in ((input_path, "input"), (model_path, "model")):
        if _same_path(output_path, other):
            raise ValueError(f"--output {output_path} resolves to the {name} file; choose a different path")
    bundle = load_model(model_path)
    df = load_transactions(input_path, require_label=False, require_time=False, features=bundle["features"])
    ranked = score_frame(bundle, df)
    if len(ranked) != len(df) or ranked[ROW].nunique() != len(df):
        raise AssertionError("scoring did not produce exactly one row per input record")

    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", dir=directory, prefix=".review_queue-", suffix=".csv", delete=False, newline="")
    try:
        with handle:
            ranked.to_csv(handle, index=False)
        os.replace(handle.name, output_path)
    except BaseException:
        if os.path.exists(handle.name):
            os.unlink(handle.name)
        raise
    model_kind = bundle.get("model_kind", BASELINE_MODEL)
    cutoff = bundle["cutoffs"][model_kind][bundle["main_budget"]]["cutoff"]
    return {
        "input": os.path.abspath(input_path),
        "input_sha256": sha256_file(input_path),
        "output": os.path.abspath(output_path),
        "run_id": bundle["run_id"],
        "rows": int(len(ranked)),
        "flagged": int(ranked["review_flag"].sum()),
        "cutoff": cutoff,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_method_table(metrics: dict, title: str) -> None:
    print(title)
    print(f"  {'method':<17}{'budget':>7}{'cutoff':>12}{'alerts':>8}{'TP':>6}{'FP':>7}{'FN':>6}{'precision':>11}{'recall':>9}")
    for method, entry in metrics.items():
        for budget, values in entry["budgets"].items():
            c = values["at_cutoff"]
            cutoff = values["frozen_cutoff"]
            print(
                f"  {method:<17}{budget:>7}{fmt_number(cutoff):>12}{c['alerts']:>8}{c['TP']:>6}{c['FP']:>7}{c['FN']:>6}"
                f"{fmt_rate(c['precision']):>11}{fmt_rate(c['recall']):>9}"
            )
        print(f"  {method:<17} AP = {fmt_number(entry['average_precision']['AP'])}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="fit a model and freeze validation cutoffs")
    p.add_argument("--input", required=True, help="historical transaction CSV; Class is optional for if_combined")
    p.add_argument("--run-dir", required=True, help="new directory for this run's artifacts")
    p.add_argument("--model-kind", choices=SCORE_COLUMNS, default=BASELINE_MODEL, help="detector to train (default isolation_forest)")
    p = sub.add_parser("evaluate", help="evaluate the frozen run on the held-out test partition")
    p.add_argument("--input", required=True, help="the same labeled CSV used for training")
    p.add_argument("--run-dir", required=True, help="run directory written by train")
    p = sub.add_parser("score", help="export a ranked review queue for a CSV")
    p.add_argument("--input", required=True, help="CSV with the saved model's features; Class and Time optional")
    p.add_argument("--model", required=True, help="model.joblib from a trusted local run")
    p.add_argument("--output", required=True, help="destination CSV for the ranked queue")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        if args.command == "train":
            record = train(args.input, args.run_dir, args.model_kind)
            split = record["split"]["partitions"]
            print(f"run_id: {record['run_id']}")
            counts = record["audit"].get("label_counts")
            frauds = counts["1"] if counts is not None else "unlabeled"
            print(f"dataset sha256: {record['dataset']['sha256']}  rows: {record['audit']['rows']}  frauds: {frauds}")
            for name, part in split.items():
                print(f"  {name:<11}{part['rows']:>8} rows  Time {part['time_range_seconds'][0]:.0f}-{part['time_range_seconds'][1]:.0f}")
            validation = record["validation"]
            if validation["metrics"] is None:
                cutoff = validation["cutoffs"][args.model_kind][MAIN_BUDGET]
                print(f"validation ({MAIN_BUDGET} target): cutoff {cutoff['cutoff']!r}, flagged {cutoff['flagged']} of {cutoff['rows']}")
            else:
                _print_method_table(validation["metrics"], "validation (cutoffs derived here):")
            print(f"artifacts: {os.path.abspath(args.run_dir)}")
        elif args.command == "evaluate":
            metrics = evaluate(args.input, args.run_dir)
            test = metrics["test"]
            print(f"run_id: {metrics['run_id']}")
            print(f"test rows: {test['rows']}  frauds: {test['class_counts']['1']}  prevalence: {test['fraud_prevalence']:.5f}")
            _print_method_table(metrics["methods"], "test (frozen validation cutoffs):")
            print(f"artifacts: {os.path.join(os.path.abspath(args.run_dir), 'metrics.json')}")
        else:
            result = score(args.input, args.model, args.output)
            print(f"scored {result['rows']} rows from {result['input']} (sha256 {result['input_sha256']})")
            print(f"flagged {result['flagged']} rows at cutoff {result['cutoff']!r} (run {result['run_id']})")
            print(f"review queue: {result['output']}")
    except (DataValidationError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
