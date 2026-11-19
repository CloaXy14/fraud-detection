"""Post-baseline experiments for plan.md section 8.

Reuses the first release's validation, chronological split, budgets, tie rule
and metrics from fraud_detection.py, which this module does not modify. Every
track writes a new run directory and never touches outputs/baseline/.

Protocol shared by all tracks
    * Same chronological partitions as the first release (60/20/20 by Time).
    * Expanding-window CV inside train + validation. The train partition is cut
      into three time blocks B1-B3 (never splitting a timestamp group) and the
      validation partition is B4. Folds: fit B1 -> score B2, fit B1-B2 -> score
      B3, fit B1-B3 (= train) -> score validation. The last fold is exactly the
      first release's train -> validation setup.
    * Candidates are ranked by mean CV average precision (AP). Selection is
      recorded before test metrics are computed.
    * Exploratory test report: the last-fold model (fit on train) scores test;
      cutoffs are frozen on validation scores with the first-release tie rule.
      The test partition was already inspected in the first release, so test
      numbers are exploratory, not a clean holdout.
    * All preprocessing (log1p(Amount), scalers, ECDFs) is fit on fit rows only.

Subcommands
    track1  Supervised classifiers and hybrid with Isolation Forest.
    track2  Isolation Forest preprocessing and tree tuning, one change at a time.
    track3  Local Outlier Factor (novelty=True), with a timed pilot for sizing.
    track4  Dense autoencoder trained on Class = 0 training rows (sklearn MLP).
    track5  Queue-volume stabilization: static vs batch top-K vs rolling cutoffs.
"""

from __future__ import annotations

import argparse
import math
import os
import resource
import shutil
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

import fraud_detection as fd
from fraud_detection import BUDGETS, FEATURES, IFOREST_PARAMS, LABEL, MAIN_BUDGET, ROW, TIME

EXPERIMENT_FORMAT_VERSION = 1
REFERENCE = "if_baseline"
CV_BLOCKS_IN_TRAIN = 3
ROLLING_WINDOW_ROWS = 20_000


class SingleClassFit(ValueError):
    """Raised when a supervised candidate's fit rows contain only one class."""


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------


def development_folds(parts: dict, blocks_in_train: int = CV_BLOCKS_IN_TRAIN):
    """Expanding-window folds inside train + validation.

    Returns ``(folds, manifest)``. ``folds`` is a list of dicts with ``name``,
    ``fit`` and ``eval`` DataFrames; the last fold fits on the full train
    partition and evaluates on validation.
    """
    train, validation = parts["train"], parts["validation"]
    times = train[TIME].to_numpy()
    n = len(train)
    bounds = [0]
    for i in range(1, blocks_in_train):
        bounds.append(fd._move_past_group(times, max(bounds[-1], math.floor(n * i / blocks_in_train))))
    bounds.append(n)
    blocks = [train.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:])]
    blocks.append(validation)
    for i, block in enumerate(blocks):
        if len(block) == 0:
            raise fd.DataValidationError(f"CV block B{i + 1} is empty; the dataset is too small for {blocks_in_train} train blocks")
    for a, b in zip(blocks[:-1], blocks[1:]):
        if not a[TIME].max() < b[TIME].min():
            raise AssertionError("CV blocks overlap in time")

    folds, manifest = [], {
        "method": (
            f"train partition cut into {blocks_in_train} blocks at floor(i/{blocks_in_train} * n) moved past the "
            "timestamp group they cut; validation is the final block. Fold k fits on blocks 1..k and scores block k+1."
        ),
        "blocks": [],
        "folds": [],
    }
    for i, block in enumerate(blocks):
        manifest["blocks"].append({
            "name": f"B{i + 1}",
            "rows": int(len(block)),
            "class_counts": fd.class_counts(block[LABEL]),
            "time_range_seconds": [float(block[TIME].min()), float(block[TIME].max())],
        })
    for k in range(1, len(blocks)):
        fit = pd.concat(blocks[:k], ignore_index=True)
        name = f"fold{k}"
        folds.append({"name": name, "fit": fit, "eval": blocks[k]})
        manifest["folds"].append({
            "name": name,
            "fit_blocks": [f"B{i + 1}" for i in range(k)],
            "eval_block": f"B{k + 1}",
            "fit_rows": int(len(fit)),
            "eval_rows": int(len(blocks[k])),
            "eval_frauds": int(blocks[k][LABEL].sum()),
        })
    return folds, manifest


# --------------------------------------------------------------------------
# Preprocessing and candidate builders
# --------------------------------------------------------------------------


def log1p_amount(X: pd.DataFrame) -> pd.DataFrame:
    """Replace Amount with log1p(Amount), keeping column order."""
    X = X.copy()
    X["Amount"] = np.log1p(X["Amount"].to_numpy(dtype=float))
    return X


PREPROCESSING = {
    "raw": "29 features as loaded",
    "log1p": "Amount -> log1p(Amount)",
    "standard": "StandardScaler fit on fit rows",
    "log1p_standard": "Amount -> log1p(Amount), then StandardScaler fit on fit rows",
    "log1p_robust": "Amount -> log1p(Amount), then RobustScaler fit on fit rows",
}


def preprocessing_steps(kind: str) -> list:
    from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

    if kind not in PREPROCESSING:
        raise ValueError(f"unknown preprocessing {kind!r}")
    steps = []
    if kind.startswith("log1p"):
        steps.append(("log1p_amount", FunctionTransformer(log1p_amount, validate=False)))
    if kind.endswith("standard"):
        steps.append(("scaler", StandardScaler()))
    elif kind.endswith("robust"):
        steps.append(("scaler", RobustScaler()))
    return steps


def _x(df: pd.DataFrame, features) -> pd.DataFrame:
    return df.loc[:, list(features)]


def _check_fit_features(model, features) -> None:
    fitted = [str(f) for f in model.feature_names_in_]
    if fitted != list(features) or any(c in fitted for c in (LABEL, TIME, ROW)):
        raise AssertionError(f"fitted features {fitted} differ from {list(features)}")


@dataclass
class Candidate:
    name: str
    family: str
    description: str
    settings: dict
    builder: Callable  # fit_df -> (scorer(df) -> np.ndarray, info dict)
    label_use: str = "none (unsupervised)"
    compares_to: str | None = REFERENCE
    track_memory: bool = False


def make_iforest(name: str, description: str, preprocess: str = "raw", drop_amount: bool = False,
                 compares_to: str | None = REFERENCE, **params) -> Candidate:
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline

    settings = dict(IFOREST_PARAMS)
    settings.update(params)
    features = [f for f in FEATURES if not (drop_amount and f == "Amount")]
    if drop_amount and preprocess.startswith("log1p"):
        raise ValueError("log1p(Amount) needs the Amount feature")

    def build(fit_df):
        model = Pipeline(preprocessing_steps(preprocess) + [("model", IsolationForest(**settings))])
        model.fit(_x(fit_df, features))
        _check_fit_features(model, features)
        return (lambda df: -model.score_samples(_x(df, features))), {"fit_rows": int(len(fit_df))}

    return Candidate(
        name, "isolation_forest", description,
        {"preprocessing": preprocess, "features": features, "params": settings,
         "score": "-IsolationForest.score_samples(X); larger is more unusual"},
        build, compares_to=compares_to,
    )


def make_amount_only() -> Candidate:
    def build(fit_df):
        return fd.amount_scores, {"fit_rows": 0}

    return Candidate("amount_only", "baseline", "Rank by raw Amount (first-release rule baseline)",
                     {"score": "Amount"}, build, compares_to=None)


SUPERVISED_KINDS = {
    "logistic_regression": "StandardScaler (fit rows) + LogisticRegression(C=1.0, lbfgs, max_iter=2000)",
    "hist_gradient_boosting": "HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=42)",
    "random_forest": "RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)",
}


def _supervised_estimator(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic_regression":
        return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(C=1.0, max_iter=2000))])
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=42)
    if kind == "random_forest":
        return RandomForestClassifier(**fd.RANDOM_FOREST_PARAMS)
    raise ValueError(f"unknown supervised model {kind!r}")


def _labels_for_fit(fit_df) -> np.ndarray:
    y = fit_df[LABEL].to_numpy()
    if len(np.unique(y)) < 2:
        raise SingleClassFit("fit rows contain a single class")
    return y


def _ecdf(reference) -> Callable:
    ordered = np.sort(np.asarray(reference, dtype=float))
    n = len(ordered)
    return lambda values: np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / n


def make_supervised(kind: str, hybrid: str | None = None) -> Candidate:
    """``hybrid``: None, "if_feature" (IF anomaly score as an extra input) or
    "rank_blend" (mean of fit-row ECDF percentiles of probability and IF score)."""
    from sklearn.base import clone
    from sklearn.ensemble import IsolationForest

    features = list(FEATURES)
    name = kind if hybrid is None else f"{kind}+{hybrid}"
    descriptions = {
        None: SUPERVISED_KINDS[kind],
        "if_feature": f"{SUPERVISED_KINDS[kind]}; adds IsolationForest anomaly_score (fit rows, first-release settings) as a 30th input",
        "rank_blend": f"0.5 * ECDF(probability) + 0.5 * ECDF(IF anomaly_score); both ECDFs fit on fit-row scores; {SUPERVISED_KINDS[kind]}",
    }

    def build(fit_df):
        y = _labels_for_fit(fit_df)
        X = _x(fit_df, features)
        estimator = clone(_supervised_estimator(kind))
        if hybrid is None:
            estimator.fit(X, y)
            _check_fit_features(estimator, features)
            return (lambda df: estimator.predict_proba(_x(df, features))[:, 1]), {"fit_rows": int(len(fit_df)), "fit_frauds": int(y.sum())}
        forest = IsolationForest(**IFOREST_PARAMS).fit(X)

        def with_if(df):
            Z = _x(df, features).copy()
            Z["if_anomaly_score"] = -forest.score_samples(_x(df, features))
            return Z

        if hybrid == "if_feature":
            estimator.fit(with_if(fit_df), y)
            _check_fit_features(estimator, features + ["if_anomaly_score"])
            return (lambda df: estimator.predict_proba(with_if(df))[:, 1]), {"fit_rows": int(len(fit_df)), "fit_frauds": int(y.sum())}
        if hybrid == "rank_blend":
            estimator.fit(X, y)
            _check_fit_features(estimator, features)
            prob_ecdf = _ecdf(estimator.predict_proba(X)[:, 1])
            if_ecdf = _ecdf(-forest.score_samples(X))

            def scorer(df):
                Xd = _x(df, features)
                return 0.5 * prob_ecdf(estimator.predict_proba(Xd)[:, 1]) + 0.5 * if_ecdf(-forest.score_samples(Xd))

            return scorer, {"fit_rows": int(len(fit_df)), "fit_frauds": int(y.sum())}
        raise ValueError(f"unknown hybrid {hybrid!r}")

    return Candidate(
        name, "supervised" if hybrid is None else "hybrid", descriptions[hybrid],
        {"model": kind, "hybrid": hybrid, "features": features + (["if_anomaly_score"] if hybrid == "if_feature" else []),
         "score": "predict_proba(X)[:, 1]" if hybrid != "rank_blend" else "ECDF rank blend"},
        build, label_use="supervised: fits on Class of fit rows",
        compares_to=REFERENCE if hybrid is None else kind,
    )


def make_lof(name: str, preprocess: str, n_neighbors: int = 20, max_fit_rows: int | None = None, seed: int = 42) -> Candidate:
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.pipeline import Pipeline

    features = list(FEATURES)

    def build(fit_df):
        sample = fit_df
        if max_fit_rows is not None and len(fit_df) > max_fit_rows:
            sample = fit_df.sample(n=max_fit_rows, random_state=seed).sort_index()
        model = Pipeline(preprocessing_steps(preprocess) + [
            ("model", LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1))])
        model.fit(_x(sample, features))
        _check_fit_features(model, features)
        return (lambda df: -model.score_samples(_x(df, features))), {"fit_rows": int(len(sample)), "fit_rows_available": int(len(fit_df))}

    return Candidate(
        name, "local_outlier_factor",
        f"{PREPROCESSING[preprocess]} + LocalOutlierFactor(n_neighbors={n_neighbors}, novelty=True)"
        + ("" if max_fit_rows is None else f"; fit on at most {max_fit_rows} rows sampled with seed {seed}"),
        {"preprocessing": preprocess, "n_neighbors": n_neighbors, "novelty": True, "max_fit_rows": max_fit_rows,
         "sample_seed": seed if max_fit_rows is not None else None,
         "score": "-LocalOutlierFactor.score_samples(X); larger is more unusual"},
        build, track_memory=True,
    )


AUTOENCODER_PARAMS = {
    "hidden_layer_sizes": (16, 8, 16),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 1e-3,
    "batch_size": 256,
    "max_iter": 100,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "tol": 1e-5,
}


def make_autoencoder(name: str, seed: int, preprocess: str = "log1p_standard") -> Candidate:
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline

    features = list(FEATURES)
    params = dict(AUTOENCODER_PARAMS, random_state=seed)

    def build(fit_df):
        legit = fit_df[fit_df[LABEL] == 0]
        steps = preprocessing_steps(preprocess)
        if not steps:
            raise ValueError("the autoencoder needs a scaler")
        pre = Pipeline(steps).fit(_x(legit, features))
        _check_fit_features(pre, features)
        Z = pre.transform(_x(legit, features))
        net = MLPRegressor(**params).fit(Z, Z)

        def scorer(df):
            Zd = pre.transform(_x(df, features))
            return np.mean((Zd - net.predict(Zd)) ** 2, axis=1)

        return scorer, {
            "fit_rows": int(len(legit)),
            "fit_rows_excluded_class_1": int(len(fit_df) - len(legit)),
            "epochs": int(net.n_iter_),
            "best_internal_validation_r2": None if net.best_validation_score_ is None else float(net.best_validation_score_),
        }

    return Candidate(
        name, "autoencoder",
        f"{PREPROCESSING[preprocess]} (fit on Class = 0 fit rows) + MLPRegressor 29-16-8-16-29, relu, linear output, seed {seed}",
        {"preprocessing": preprocess, "params": {**params, "hidden_layer_sizes": list(params["hidden_layer_sizes"])},
         "score": "mean squared reconstruction error per row in the scaled space"},
        build, label_use="semi-supervised: fit rows filtered to Class = 0; internal early-stopping split is a random 10% of those rows",
    )


# --------------------------------------------------------------------------
# Running and summarising candidates
# --------------------------------------------------------------------------


def ranking_metrics(labels, scores, row_numbers) -> dict:
    labels = np.asarray(labels)
    return {
        "rows": int(len(labels)),
        "frauds": int(labels.sum()),
        "average_precision": fd.average_precision(labels, scores),
        "top_k": {budget: fd.top_k_metrics(labels, scores, row_numbers, rate) for budget, rate in BUDGETS.items()},
    }


def top_tail_profile(part: pd.DataFrame, scores, rate: str = BUDGETS[MAIN_BUDGET]) -> dict:
    """Amounts of false and true positives in the fixed top-K of a partition."""
    labels = part[LABEL].to_numpy().astype(bool)
    amounts = part["Amount"].to_numpy()
    k = fd.budget_size(rate, len(part))
    top = fd.ranking_order(scores, part[ROW].to_numpy())[:k]
    fp, tp = top[~labels[top]], top[labels[top]]

    def amount_summary(idx):
        if len(idx) == 0:
            return None
        values = amounts[idx]
        return {"median": float(np.median(values)), "p75": float(np.quantile(values, 0.75)),
                "max": float(values.max()), "share_at_least_1000": float(np.mean(values >= 1000))}

    return {"rate": rate, "K": k, "TP": int(len(tp)), "FP": int(len(fp)),
            "false_positive_amounts": amount_summary(fp), "true_positive_amounts": amount_summary(tp)}


def _timed(fn, track_memory: bool):
    if track_memory:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1] if track_memory else None
    finally:
        if track_memory:
            tracemalloc.stop()
    return result, elapsed, peak


def run_candidate(candidate: Candidate, folds: list, test: pd.DataFrame, log=print) -> dict:
    """Fit on each fold; the last fold's model also scores the test partition."""
    out = {"folds": {}, "validation_scores": None, "test_scores": None, "test_score_seconds": None}
    for i, fold in enumerate(folds):
        entry = {}
        try:
            (scorer, info), fit_s, fit_peak = _timed(lambda: candidate.builder(fold["fit"]), candidate.track_memory)
        except SingleClassFit as exc:
            out["folds"][fold["name"]] = {"skipped": str(exc)}
            log(f"  {candidate.name} {fold['name']}: skipped ({exc})")
            continue
        scores, score_s, score_peak = _timed(lambda: scorer(fold["eval"]), candidate.track_memory)
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"{candidate.name} produced non-finite scores on {fold['name']}")
        entry.update(ranking_metrics(fold["eval"][LABEL].to_numpy(), scores, fold["eval"][ROW].to_numpy()))
        entry.update(fit_info=info, fit_seconds=fit_s, score_seconds=score_s,
                     score_microseconds_per_row=1e6 * score_s / len(fold["eval"]))
        if candidate.track_memory:
            entry["peak_traced_memory_mb"] = {"fit": fit_peak / 2**20, "score": score_peak / 2**20}
        out["folds"][fold["name"]] = entry
        ap = entry["average_precision"]["AP"]
        log(f"  {candidate.name} {fold['name']}: AP={fd.fmt_number(ap)} fit={fit_s:.1f}s score={score_s:.1f}s")
        if i == len(folds) - 1:
            out["validation_scores"] = scores
            started = time.perf_counter()
            out["test_scores"] = scorer(test)
            out["test_score_seconds"] = time.perf_counter() - started
    return out


def cv_summary(fold_results: dict) -> dict:
    done = [f for f in fold_results.values() if "skipped" not in f]
    aps = [f["average_precision"]["AP"] for f in done if f["average_precision"]["AP"] is not None]
    summary = {
        "folds_used": len(done),
        "folds_skipped": len(fold_results) - len(done),
        "mean_AP": float(np.mean(aps)) if aps else None,
        "std_AP": float(np.std(aps)) if aps else None,
        "fold_AP": [None if "skipped" in f else f["average_precision"]["AP"] for f in fold_results.values()],
        "pooled_top_k": {},
        "mean_fit_seconds": float(np.mean([f["fit_seconds"] for f in done])) if done else None,
        "mean_score_microseconds_per_row": float(np.mean([f["score_microseconds_per_row"] for f in done])) if done else None,
    }
    for budget in BUDGETS:
        tp = sum(f["top_k"][budget]["TP_in_top_K"] for f in done)
        k = sum(f["top_k"][budget]["K"] for f in done)
        frauds = sum(f["frauds"] for f in done)
        summary["pooled_top_k"][budget] = {"K_total": k, "TP_total": tp, "precision": fd._ratio(tp, k), "recall": fd._ratio(tp, frauds)}
    return summary


def select_best(summaries: dict, names=None):
    names = list(summaries) if names is None else list(names)
    eligible = [n for n in names if summaries[n]["mean_AP"] is not None]
    if not eligible:
        return None
    return max(eligible, key=lambda n: summaries[n]["mean_AP"])  # first wins ties (stable order)


def test_report(validation: pd.DataFrame, test: pd.DataFrame, name: str, val_scores, test_scores) -> dict:
    cutoffs = {b: fd.select_cutoff(val_scores, r) for b, r in BUDGETS.items()}
    metrics = fd.evaluate_methods(test[LABEL].to_numpy(), test[ROW].to_numpy(), {name: test_scores}, {name: cutoffs})[name]
    metrics["validation_cutoffs"] = cutoffs
    metrics["top_tail_profile"] = top_tail_profile(test, test_scores)
    if fd.average_precision(test[LABEL].to_numpy(), test_scores)["available"]:
        metrics["score_quantiles"] = {"validation": fd.quantiles(val_scores), "test": fd.quantiles(test_scores)}
    return metrics


def calibration_report(labels, probabilities) -> dict:
    from sklearn.metrics import brier_score_loss

    labels = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    k = fd.budget_size(BUDGETS[MAIN_BUDGET], len(p))
    top = np.argsort(-p, kind="mergesort")[:k]
    return {
        "brier_score": float(brier_score_loss(labels, p)) if len(np.unique(labels)) > 1 else None,
        "mean_probability": float(p.mean()),
        "observed_fraud_rate": float(labels.mean()),
        f"top_{MAIN_BUDGET}_mean_probability": float(p[top].mean()) if k else None,
        f"top_{MAIN_BUDGET}_observed_precision": float(labels[top].mean()) if k else None,
    }


# --------------------------------------------------------------------------
# Run directories
# --------------------------------------------------------------------------


@dataclass
class Context:
    input_path: str
    run_dir: str
    started: float
    df: pd.DataFrame
    data_hash: str
    parts: dict
    split_manifest: dict
    folds: list
    folds_manifest: dict
    candidates: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


def prepare(input_path: str, run_dir: str, log=print) -> Context:
    started = time.perf_counter()
    run_dir = os.path.abspath(run_dir)
    if os.path.exists(run_dir):
        raise FileExistsError(f"{run_dir} already exists; choose a new --run-dir so earlier runs are preserved")
    df = fd.load_transactions(input_path, require_label=True, require_time=True)
    data_hash = fd.sha256_file(input_path)
    parts, manifest = fd.chronological_split(df)
    for name in fd.PARTITIONS:
        manifest["partitions"][name]["class_counts"] = fd.class_counts(parts[name][LABEL])
    folds, folds_manifest = development_folds(parts)
    log(f"loaded {len(df)} rows; folds: " + ", ".join(f"{f['name']} fit={len(f['fit'])} eval={len(f['eval'])}" for f in folds))
    return Context(input_path, run_dir, started, df, data_hash, parts, manifest, folds, folds_manifest)


def evaluate_all(ctx: Context, candidates: list, log=print) -> None:
    for candidate in candidates:
        if candidate.name in ctx.results:
            continue
        log(f"candidate {candidate.name}: {candidate.description}")
        ctx.candidates[candidate.name] = candidate
        ctx.results[candidate.name] = run_candidate(candidate, ctx.folds, ctx.parts["test"], log=log)


def build_metrics(ctx: Context, selection: dict, extra: dict | None = None) -> dict:
    """CV summaries, last-fold validation metrics and exploratory test metrics."""
    validation, test = ctx.parts["validation"], ctx.parts["test"]
    summaries = {n: cv_summary(r["folds"]) for n, r in ctx.results.items()}
    metrics = {
        "protocol": {
            "selection_metric": "mean average precision across expanding-window CV folds",
            "test_status": "exploratory: the test partition was inspected in the first release",
            "cutoff_rule": "first-release rule: K = floor(rate * validation rows), whole tied groups, flag score >= cutoff",
        },
        "cv_summary": summaries,
        "selection": selection,
        "cv_folds": {n: r["folds"] for n, r in ctx.results.items()},
        "validation_last_fold": {},
        "test_exploratory": {},
    }
    for name, result in ctx.results.items():
        if result["validation_scores"] is None:
            metrics["validation_last_fold"][name] = {"skipped": "last fold was skipped"}
            metrics["test_exploratory"][name] = {"skipped": "last fold was skipped"}
            continue
        last = result["folds"][ctx.folds[-1]["name"]]
        metrics["validation_last_fold"][name] = {
            "average_precision": last["average_precision"],
            "top_k": last["top_k"],
            "top_tail_profile": top_tail_profile(validation, result["validation_scores"]),
        }
        report = test_report(validation, test, name, result["validation_scores"], result["test_scores"])
        report["test_score_seconds"] = result["test_score_seconds"]
        if ctx.candidates[name].settings.get("score") == "predict_proba(X)[:, 1]":
            report["calibration"] = {
                "validation": calibration_report(validation[LABEL].to_numpy(), result["validation_scores"]),
                "test": calibration_report(test[LABEL].to_numpy(), result["test_scores"]),
            }
        metrics["test_exploratory"][name] = report
    if extra:
        metrics.update(extra)
    return metrics


def publish(ctx: Context, track: str, metrics: dict, settings: dict, plot_names: list | None = None, log=print) -> dict:
    run_dir = ctx.run_dir
    run_id = f"{os.path.basename(run_dir)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_record = {
        "run_id": run_id,
        "format_version": EXPERIMENT_FORMAT_VERSION,
        "track": track,
        "created_utc": fd._utc_now(),
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv) if sys.argv else None,
        "reference_run": "baseline-20261119T053308Z (outputs/baseline, unchanged)",
        "dataset": {"path": os.path.abspath(ctx.input_path), "sha256": ctx.data_hash, "rows": int(len(ctx.df))},
        "split": ctx.split_manifest,
        "cv": ctx.folds_manifest,
        "candidates": {
            n: {"family": c.family, "description": c.description, "settings": c.settings,
                "label_use": c.label_use, "compares_to": c.compares_to}
            for n, c in ctx.candidates.items()
        },
        "settings": settings,
        "versions": fd.environment_versions(),
        "cpu_count": os.cpu_count(),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10),
        "duration_seconds": None,
    }
    metrics = {"run_id": run_id, **metrics}
    parent = os.path.dirname(run_dir) or "."
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{os.path.basename(run_dir)}.staging-", dir=parent)
    try:
        validation, test = ctx.parts["validation"], ctx.parts["test"]
        names = plot_names if plot_names is not None else list(ctx.results)
        for part_name, part, key, section in (
            ("validation", validation, "validation_scores", "validation_last_fold"),
            ("test", test, "test_scores", "test_exploratory"),
        ):
            labels = part[LABEL].to_numpy()
            curves = []
            for n in names:
                result = ctx.results.get(n)
                ap = metrics[section].get(n, {}).get("average_precision", {}).get("AP")
                if result is not None and result[key] is not None and ap is not None:
                    curves.append((n, labels, result[key], ap))
            if curves:
                title = "Validation (last CV fold)" if part_name == "validation" else "Test (exploratory)"
                fd.plot_precision_recall(curves, float(labels.mean()), f"{track}: {title} precision-recall",
                                         os.path.join(staging, f"{part_name}_precision_recall.png"))
        run_record["duration_seconds"] = time.perf_counter() - ctx.started
        metrics["duration_seconds"] = run_record["duration_seconds"]
        fd.write_json(os.path.join(staging, "run.json"), run_record)
        fd.write_json(os.path.join(staging, "metrics.json"), metrics)
        fd._publish_dir(staging, run_dir)
    except BaseException:
        if os.path.isdir(staging) and not os.path.exists(run_dir):
            shutil.rmtree(staging, ignore_errors=True)
        raise
    log(f"wrote {run_dir} (run {run_id})")
    return {"run_id": run_id, "run_dir": run_dir, "metrics": metrics, "run": run_record}


def print_table(metrics: dict, names=None) -> None:
    names = list(metrics["cv_summary"]) if names is None else names
    print(f"{'candidate':42s} {'CV AP':>13s} {'val AP':>7s} {'test AP':>7s} {'test R@0.5%cut':>14s} {'test R@1%cut':>12s} {'test R@top1%':>12s} {'fit s':>6s}")
    for n in names:
        s = metrics["cv_summary"][n]
        v = metrics["validation_last_fold"][n].get("average_precision", {}).get("AP")
        t = metrics["test_exploratory"][n]
        cv = "N/A" if s["mean_AP"] is None else f"{s['mean_AP']:.4f}±{s['std_AP']:.4f}"
        if "skipped" in t:
            print(f"{n:42s} {cv:>13s} skipped")
            continue
        b05, b1 = t["budgets"]["0.5%"], t["budgets"]["1.0%"]
        print(f"{n:42s} {cv:>13s} {fd.fmt_number(v):>7s} {fd.fmt_number(t['average_precision']['AP']):>7s} "
              f"{fd.fmt_rate(b05['at_cutoff']['recall']):>14s} {fd.fmt_rate(b1['at_cutoff']['recall']):>12s} "
              f"{fd.fmt_rate(b1['fixed_size_ranking']['recall_at_K']):>12s} {fd.fmt_number(s['mean_fit_seconds'], 1):>6s}")


# --------------------------------------------------------------------------
# Tracks 1-4
# --------------------------------------------------------------------------


def reference_candidates() -> list:
    return [make_iforest(REFERENCE, "First-release Isolation Forest (raw 29 features, default settings)", compares_to=None),
            make_amount_only()]


def track1(input_path: str, run_dir: str, log=print) -> dict:
    ctx = prepare(input_path, run_dir, log)
    evaluate_all(ctx, reference_candidates() + [make_supervised(k) for k in SUPERVISED_KINDS], log)
    summaries = {n: cv_summary(r["folds"]) for n, r in ctx.results.items()}
    best_supervised = select_best(summaries, SUPERVISED_KINDS)
    if best_supervised is not None:
        evaluate_all(ctx, [make_supervised(best_supervised, "if_feature"), make_supervised(best_supervised, "rank_blend")], log)
    summaries = {n: cv_summary(r["folds"]) for n, r in ctx.results.items()}
    selection = {
        "rule": "Stage 1: best supervised model by mean CV AP; stage 2: hybrids built on that model; final pick = best mean CV AP overall.",
        "best_supervised_stage1": best_supervised,
        "selected": select_best(summaries),
        "selected_before_test_metrics": True,
    }
    metrics = build_metrics(ctx, selection)
    print_table(metrics)
    return publish(ctx, "track1-supervised-hybrid", metrics, {"supervised_models": SUPERVISED_KINDS}, log=log)


def track2(input_path: str, run_dir: str, log=print) -> dict:
    ctx = prepare(input_path, run_dir, log)
    reference = reference_candidates()
    dimensions = {
        "amount_handling": {
            "log1p": dict(preprocess="log1p"),
            "log1p_robust": dict(preprocess="log1p_robust"),
            "log1p_standard": dict(preprocess="log1p_standard"),
            "drop_amount": dict(drop_amount=True),
        },
        "max_samples": {"max_samples_2048": dict(max_samples=2048), "max_samples_4096": dict(max_samples=4096)},
        "n_estimators": {"n_estimators_300": dict(n_estimators=300)},
        "max_features": {"max_features_0.75": dict(max_features=0.75), "max_features_0.5": dict(max_features=0.5)},
    }
    singles = []
    for dim, options in dimensions.items():
        for option, kwargs in options.items():
            singles.append(make_iforest(f"if_{option}", f"Isolation Forest, one change vs {REFERENCE}: {option}", **kwargs))
    evaluate_all(ctx, reference + singles, log)
    summaries = {n: cv_summary(r["folds"]) for n, r in ctx.results.items()}
    base_ap = summaries[REFERENCE]["mean_AP"]
    combined_kwargs, chosen = {}, {}
    for dim, options in dimensions.items():
        best = select_best(summaries, [f"if_{o}" for o in options])
        if best is not None and summaries[best]["mean_AP"] > base_ap:
            chosen[dim] = best
            combined_kwargs.update(options[best[len("if_"):]])
        else:
            chosen[dim] = None
    if len([c for c in chosen.values() if c]) >= 2:
        evaluate_all(ctx, [make_iforest("if_combined", "Isolation Forest stacking the best CV change per dimension: "
                                        + ", ".join(c for c in chosen.values() if c), **combined_kwargs)], log)
    summaries = {n: cv_summary(r["folds"]) for n, r in ctx.results.items()}
    unsupervised = [n for n in ctx.results if n != "amount_only"]
    selection = {
        "rule": "Each change is a single edit to the first-release settings. Per dimension, keep the best change only if its mean CV AP beats if_baseline; stack kept changes into if_combined when two or more qualify. Final pick = best mean CV AP among Isolation Forest variants.",
        "kept_changes_per_dimension": chosen,
        "selected": select_best(summaries, unsupervised),
        "selected_before_test_metrics": True,
    }
    metrics = build_metrics(ctx, selection)
    print_table(metrics)
    plot = [REFERENCE, "amount_only", selection["selected"]] + (["if_combined"] if "if_combined" in ctx.results else [])
    return publish(ctx, "track2-isolation-forest-tuning", metrics, {"dimensions": {d: list(o) for d, o in dimensions.items()}},
                   plot_names=list(dict.fromkeys(plot)), log=log)


def lof_pilot(fit_df: pd.DataFrame, eval_df: pd.DataFrame, budget_seconds: float, sizes=(10_000, 20_000), seed: int = 42, log=print) -> dict:
    """Time LOF fit+score on training-only samples and extrapolate quadratically."""
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    timings = []
    probe = eval_df.iloc[: min(len(eval_df), 5_000)]
    for size in sizes:
        size = min(size, len(fit_df))
        sample = fit_df.sample(n=size, random_state=seed)
        scaler = StandardScaler().fit(_x(sample, FEATURES))
        started = time.perf_counter()
        model = LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(scaler.transform(_x(sample, FEATURES)))
        fit_s = time.perf_counter() - started
        started = time.perf_counter()
        model.score_samples(scaler.transform(_x(probe, FEATURES)))
        score_s = time.perf_counter() - started
        timings.append({"fit_rows": size, "fit_seconds": fit_s, "score_rows": int(len(probe)), "score_seconds": score_s})
        log(f"  LOF pilot n={size}: fit {fit_s:.1f}s, score {len(probe)} rows {score_s:.1f}s")
    largest = timings[-1]
    fit_coef = largest["fit_seconds"] / largest["fit_rows"] ** 2
    score_coef = largest["score_seconds"] / (largest["fit_rows"] * largest["score_rows"])
    full_n, eval_n = len(fit_df), len(eval_df)
    estimate = fit_coef * full_n**2 + score_coef * full_n * eval_n
    if estimate <= budget_seconds:
        max_rows = None
    else:
        # Solve fit_coef*n^2 + score_coef*n*eval_n = budget for n, rounded down to 10k rows.
        a, b = fit_coef, score_coef * eval_n
        n = (-b + math.sqrt(b * b + 4 * a * budget_seconds)) / (2 * a)
        max_rows = max(10_000, int(n // 10_000) * 10_000)
    return {"seed": seed, "timings": timings, "largest_fit_rows": full_n,
            "estimated_full_fit_plus_score_seconds": estimate, "budget_seconds_per_fold": budget_seconds,
            "max_fit_rows": max_rows,
            "decision": "full fit rows" if max_rows is None else f"subsample fit rows to {max_rows} (seed {seed})"}


def track3(input_path: str, run_dir: str, budget_seconds: float = 600.0, log=print) -> dict:
    ctx = prepare(input_path, run_dir, log)
    pilot = lof_pilot(ctx.folds[-1]["fit"], ctx.folds[-1]["eval"], budget_seconds, log=log)
    log(f"LOF pilot: estimated {pilot['estimated_full_fit_plus_score_seconds']:.0f}s per full fold -> {pilot['decision']}")
    lofs = [make_lof("lof_standard_k20", "standard", max_fit_rows=pilot["max_fit_rows"]),
            make_lof("lof_log1p_standard_k20", "log1p_standard", max_fit_rows=pilot["max_fit_rows"])]
    evaluate_all(ctx, reference_candidates() + lofs, log)
    summaries = {n: cv_summary(r["folds"]) for n, r in ctx.results.items()}
    selection = {"rule": "best mean CV AP among LOF variants; compared with if_baseline on the same folds",
                 "selected": select_best(summaries, [c.name for c in lofs]), "selected_before_test_metrics": True}
    metrics = build_metrics(ctx, selection, {"lof_pilot": pilot})
    print_table(metrics)
    return publish(ctx, "track3-local-outlier-factor", metrics, {"lof_pilot": pilot}, log=log)


def track4(input_path: str, run_dir: str, seeds=(0, 1, 2), log=print) -> dict:
    ctx = prepare(input_path, run_dir, log)
    autoencoders = [make_autoencoder(f"autoencoder_log1p_standard_seed{s}", s) for s in seeds]
    autoencoders.append(make_autoencoder(f"autoencoder_standard_seed{seeds[0]}", seeds[0], preprocess="standard"))
    evaluate_all(ctx, reference_candidates() + autoencoders, log)
    summaries = {n: cv_summary(r["folds"]) for n, r in ctx.results.items()}
    seed_names = [a.name for a in autoencoders[: len(seeds)]]
    seed_aps = [summaries[n]["mean_AP"] for n in seed_names]
    selection = {"rule": "best mean CV AP among autoencoder variants",
                 "selected": select_best(summaries, [a.name for a in autoencoders]), "selected_before_test_metrics": True}
    metrics = build_metrics(ctx, selection)
    test_aps = [metrics["test_exploratory"][n]["average_precision"]["AP"] for n in seed_names]
    metrics["seed_sensitivity"] = {
        "seeds": list(seeds),
        "cv_mean_AP": {"values": seed_aps, "mean": float(np.mean(seed_aps)), "std": float(np.std(seed_aps))},
        "test_AP": {"values": test_aps, "mean": float(np.mean(test_aps)), "std": float(np.std(test_aps))},
        "epochs": {n: [f.get("fit_info", {}).get("epochs") for f in metrics["cv_folds"][n].values()] for n in seed_names},
    }
    print_table(metrics)
    return publish(ctx, "track4-autoencoder", metrics,
                   {"autoencoder_params": {**AUTOENCODER_PARAMS, "hidden_layer_sizes": list(AUTOENCODER_PARAMS["hidden_layer_sizes"])},
                    "seeds": list(seeds)}, log=log)


# --------------------------------------------------------------------------
# Track 5: queue-volume stabilization
# --------------------------------------------------------------------------


def hourly_batches(times) -> np.ndarray:
    return (np.asarray(times, dtype=float) // 3600).astype(np.int64)


def batch_top_k_flags(scores, row_numbers, batches, k_per_batch) -> np.ndarray:
    """Flag the top K rows of each batch (score desc, row_number asc).

    ``k_per_batch``: callable(batch_rows) -> K for that batch.
    """
    scores, row_numbers, batches = np.asarray(scores, dtype=float), np.asarray(row_numbers), np.asarray(batches)
    flags = np.zeros(len(scores), dtype=np.int64)
    for batch in np.unique(batches):
        idx = np.flatnonzero(batches == batch)
        k = min(int(k_per_batch(len(idx))), len(idx))
        if k > 0:
            order = fd.ranking_order(scores[idx], row_numbers[idx])[:k]
            flags[idx[order]] = 1
    return flags


def rolling_cutoffs(reference_scores, apply_scores, rate: str, window: int) -> np.ndarray:
    """Per-row cutoff = (1 - rate) quantile of the previous ``window`` unlabeled scores.

    History is the reference scores followed by the scores of earlier rows in
    the apply block; the current row is never in its own window.
    """
    reference_scores = np.asarray(reference_scores, dtype=float)
    if len(reference_scores) < window:
        raise ValueError(f"reference block has {len(reference_scores)} rows, fewer than the window of {window}")
    history = pd.Series(np.concatenate([reference_scores, np.asarray(apply_scores, dtype=float)]))
    q = history.rolling(window, min_periods=window).quantile(1 - float(rate)).shift(1)
    return q.to_numpy()[len(reference_scores):]


def stability_metrics(labels, flags, batches, rate: str) -> dict:
    labels, flags, batches = np.asarray(labels).astype(bool), np.asarray(flags).astype(bool), np.asarray(batches)
    target = float(rate)
    hourly = []
    for batch in np.unique(batches):
        idx = batches == batch
        hourly.append({"hour": int(batch), "rows": int(idx.sum()), "alerts": int(flags[idx].sum()),
                       "alert_rate": float(flags[idx].mean())})
    rates = np.array([h["alert_rate"] for h in hourly])
    weights = np.array([h["rows"] for h in hourly], dtype=float)
    base = fd.confusion_metrics(labels, flags)
    return {
        "target_alerts": fd.budget_size(rate, len(labels)),
        "alerts": base["alerts"],
        "alerts_vs_target": fd._ratio(base["alerts"], fd.budget_size(rate, len(labels))),
        "TP": base["TP"], "frauds": base["positive_support"],
        "precision": base["precision"], "recall": base["recall"],
        "hours": len(hourly),
        "hourly_alert_rate_std": float(np.std(rates)),
        "hourly_alert_rate_weighted_mean_abs_error": float(np.average(np.abs(rates - target), weights=weights)),
        "hourly_alert_rate_max_abs_error": float(np.max(np.abs(rates - target))),
        "hourly": hourly,
    }


def queue_rules(reference: pd.DataFrame, ref_scores, apply: pd.DataFrame, apply_scores, rate: str, window: int) -> dict:
    labels = apply[LABEL].to_numpy()
    rows = apply[ROW].to_numpy()
    batches = hourly_batches(apply[TIME])
    ref_hours = len(np.unique(hourly_batches(reference[TIME])))
    fixed_k = math.floor(float(rate) * len(reference) / ref_hours)
    cutoff = fd.select_cutoff(ref_scores, rate)
    rolling = rolling_cutoffs(ref_scores, apply_scores, rate, window)
    rules = {
        "static_cutoff": (fd.apply_cutoff(apply_scores, cutoff["cutoff"]),
                          f"first-release rule: cutoff frozen on the reference block ({cutoff['cutoff']!r})"),
        "batch_top_k_proportional": (batch_top_k_flags(apply_scores, rows, batches, lambda n: fd.budget_size(rate, n)),
                                     "hourly batches; K = floor(rate * batch rows)"),
        "batch_top_k_fixed_capacity": (batch_top_k_flags(apply_scores, rows, batches, lambda n: fixed_k),
                                       f"hourly batches; K = {fixed_k} per batch = floor(rate * reference rows / reference hours)"),
        "rolling_quantile": ((np.asarray(apply_scores) >= rolling).astype(np.int64),
                             f"flag score >= (1 - rate) quantile of the previous {window} unlabeled scores (reference tail, then earlier apply rows)"),
    }
    return {name: {"rule": desc, **stability_metrics(labels, flags, batches, rate)} for name, (flags, desc) in rules.items()}


def track5(input_path: str, run_dir: str, models=(REFERENCE,), window: int = ROLLING_WINDOW_ROWS, log=print) -> dict:
    ctx = prepare(input_path, run_dir, log)
    tuned = make_iforest("if_combined", "Tuned Isolation Forest (Track 2 recipe)", drop_amount=True, **fd.IF_COMBINED_PARAMS)
    registry = {c.name: c for c in reference_candidates() + [tuned] + [make_supervised(k) for k in SUPERVISED_KINDS]}
    unknown = [m for m in models if m not in registry]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {sorted(registry)}")
    blocks = [ctx.folds[0]["fit"]] + [f["eval"] for f in ctx.folds]  # B1..B4
    windows = []
    for k in range(1, len(blocks) - 1):  # fit B1..Bk, reference B(k+1), apply B(k+2)
        windows.append({"name": f"dev_window{k}", "fit": pd.concat(blocks[:k], ignore_index=True),
                        "reference": blocks[k], "apply": blocks[k + 1], "status": "development"})
    windows.append({"name": "test", "fit": ctx.parts["train"], "reference": ctx.parts["validation"],
                    "apply": ctx.parts["test"], "status": "exploratory"})
    results = {}
    for model in models:
        candidate = registry[model]
        ctx.candidates[model] = candidate
        results[model] = {}
        for w in windows:
            try:
                scorer, _ = candidate.builder(w["fit"])
            except SingleClassFit as exc:
                results[model][w["name"]] = {"skipped": str(exc)}
                continue
            ref_scores, apply_scores = scorer(w["reference"]), scorer(w["apply"])
            entry = {"status": w["status"], "fit_rows": int(len(w["fit"])), "reference_rows": int(len(w["reference"])),
                     "apply_rows": int(len(w["apply"])), "apply_frauds": int(w["apply"][LABEL].sum()), "budgets": {}}
            for budget in BUDGETS:
                entry["budgets"][budget] = queue_rules(w["reference"], ref_scores, w["apply"], apply_scores, BUDGETS[budget], window)
            results[model][w["name"]] = entry
            for budget in BUDGETS:
                line = ", ".join(f"{r}: {v['alerts']}/{v['target_alerts']} alerts R={fd.fmt_rate(v['recall'])} hourlyMAE={v['hourly_alert_rate_weighted_mean_abs_error']:.4f}"
                                 for r, v in entry["budgets"][budget].items())
                log(f"  {model} {w['name']} {budget}: {line}")
    metrics = {
        "protocol": {
            "windows": "Fit on blocks before the reference block; freeze or seed rules on the reference block's unlabeled scores; measure on the next block. dev windows use B1..B4 inside train + validation; the test window is exploratory.",
            "batches": "hourly by floor(Time / 3600)",
            "rolling_window_rows": window,
        },
        "windows": {w["name"]: {"fit_rows": int(len(w["fit"])), "reference_rows": int(len(w["reference"])),
                                "apply_rows": int(len(w["apply"])), "status": w["status"]} for w in windows},
        "results": results,
    }
    return publish(ctx, "track5-queue-stability", metrics, {"models": list(models), "rolling_window_rows": window}, plot_names=[], log=log)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("track1", "supervised classifiers and hybrids"),
        ("track2", "Isolation Forest tuning"),
        ("track3", "Local Outlier Factor"),
        ("track4", "autoencoder"),
        ("track5", "queue-volume stabilization"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--input", required=True, help="labeled CSV (data/creditcard.csv)")
        p.add_argument("--run-dir", required=True, help="new directory for this experiment's artifacts")
        if name == "track3":
            p.add_argument("--budget-seconds", type=float, default=600.0, help="per-fold LOF fit+score time budget used to size the training sample")
        if name == "track4":
            p.add_argument("--seeds", default="0,1,2", help="comma-separated autoencoder seeds")
        if name == "track5":
            p.add_argument("--models", default=REFERENCE, help="comma-separated candidate names (if_baseline, if_combined, logistic_regression, hist_gradient_boosting, random_forest)")
            p.add_argument("--window", type=int, default=ROLLING_WINDOW_ROWS, help="rolling cutoff window in rows")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    try:
        if args.command == "track1":
            track1(args.input, args.run_dir)
        elif args.command == "track2":
            track2(args.input, args.run_dir)
        elif args.command == "track3":
            track3(args.input, args.run_dir, budget_seconds=args.budget_seconds)
        elif args.command == "track4":
            track4(args.input, args.run_dir, seeds=tuple(int(s) for s in args.seeds.split(",")))
        elif args.command == "track5":
            track5(args.input, args.run_dir, models=tuple(m.strip() for m in args.models.split(",") if m.strip()), window=args.window)
    except (fd.DataValidationError, FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
