"""Unit and integration tests for fraud_detection.py.

Covers plan.md §7 verification checklist and milestones 1-5:
  - Schema validation, reordered columns, invalid inputs, path collisions
  - Chronological splitting with shuffled rows and repeated timestamps
  - Feature isolation (no Class, Time, or row_number leakage) and score direction
  - Cutoff selection, ties, exact boundary equality, empty queues
  - Hand-calculated metrics and degenerate (no-alert / no-fraud / single-class) cases
  - Model serialization parity, evaluation guards, and batch scoring round-trip
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

import fraud_detection as fd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(TESTS_DIR, "fixtures")
LABELED_FIXTURE = os.path.join(FIXTURES_DIR, "labeled_small.csv")
UNLABELED_FIXTURE = os.path.join(FIXTURES_DIR, "unlabeled_small.csv")


def _make_minimal_frame(rows: int = 30, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"Time": np.arange(rows, dtype=float)}
    for i in range(1, 29):
        data[f"V{i}"] = np.round(rng.normal(0.0, 1.0, size=rows), 6)
    data["Amount"] = np.round(rng.uniform(1.0, 500.0, size=rows), 2)
    data["Class"] = np.zeros(rows, dtype=int)
    for idx in (2, 19, 27):
        if idx < rows:
            data["Class"][idx] = 1
    return pd.DataFrame(data)


class TestSchemaAndValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="fraud-test-schema-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_csv(self, name: str, df: pd.DataFrame) -> str:
        path = os.path.join(self.tmpdir, name)
        df.to_csv(path, index=False)
        return path

    def test_reordered_feature_columns_produce_identical_scores(self) -> None:
        run_dir = os.path.join(self.tmpdir, "run")
        fd.train(LABELED_FIXTURE, run_dir)
        model_path = os.path.join(run_dir, "model.joblib")

        canonical = pd.read_csv(UNLABELED_FIXTURE)
        reordered_cols = ["Amount"] + [f"V{i}" for i in range(28, 0, -1)] + ["Time"]
        reordered_path = self._write_csv("reordered.csv", canonical[reordered_cols])

        q_canon = os.path.join(self.tmpdir, "q_canon.csv")
        q_reord = os.path.join(self.tmpdir, "q_reord.csv")
        fd.score(UNLABELED_FIXTURE, model_path, q_canon)
        fd.score(reordered_path, model_path, q_reord)

        df_canon = pd.read_csv(q_canon)
        df_reord = pd.read_csv(q_reord)
        np.testing.assert_array_equal(df_canon["row_number"].to_numpy(), df_reord["row_number"].to_numpy())
        np.testing.assert_allclose(df_canon["anomaly_score"].to_numpy(), df_reord["anomaly_score"].to_numpy(), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(df_canon["review_flag"].to_numpy(), df_reord["review_flag"].to_numpy())

    def test_missing_columns_rejected(self) -> None:
        df = _make_minimal_frame().drop(columns=["V14", "Amount"])
        path = self._write_csv("missing.csv", df)
        with self.assertRaises(fd.DataValidationError) as ctx:
            fd.load_transactions(path, require_label=True, require_time=True)
        msg = str(ctx.exception)
        self.assertIn("V14", msg)
        self.assertIn("Amount", msg)

    def test_duplicate_headers_rejected(self) -> None:
        df = _make_minimal_frame(rows=5)
        path = os.path.join(self.tmpdir, "dup_header.csv")
        cols = list(df.columns) + ["Amount"]
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(",".join(cols) + "\n")
            for _, row in df.iterrows():
                vals = [str(row[c]) for c in df.columns] + [str(row["Amount"])]
                handle.write(",".join(vals) + "\n")
        with self.assertRaises(fd.DataValidationError) as ctx:
            fd.load_transactions(path, require_label=True, require_time=True)
        self.assertIn("duplicate header", str(ctx.exception))
        self.assertIn("Amount", str(ctx.exception))

    def test_invalid_values_report_columns_and_row_numbers(self) -> None:
        df = _make_minimal_frame(rows=10).astype(str)
        df.loc[0, "V1"] = ""          # row 1 missing
        df.loc[1, "V2"] = "notanumber"  # row 2 nonnumeric
        df.loc[2, "V3"] = "NaN"       # row 3 NaN text
        df.loc[3, "V4"] = "inf"       # row 4 infinite
        df.loc[4, "V5"] = "-inf"      # row 5 infinite
        df.loc[5, "Time"] = "-5"      # row 6 negative Time
        df.loc[6, "Amount"] = "-1.25"  # row 7 negative Amount
        df.loc[7, "Class"] = "2"      # row 8 invalid Class
        path = self._write_csv("invalid.csv", df)

        with self.assertRaises(fd.DataValidationError) as ctx:
            fd.load_transactions(path, require_label=True, require_time=True)
        msg = str(ctx.exception)
        for expected in (
            "column 'V1'", "row(s) 1",
            "column 'V2'", "row(s) 2",
            "column 'V3'", "row(s) 3",
            "column 'V4'", "row(s) 4",
            "column 'V5'", "row(s) 5",
            "column 'Time'", "row(s) 6",
            "column 'Amount'", "row(s) 7",
            "column 'Class'", "row(s) 8",
        ):
            self.assertIn(expected, msg)

    def test_empty_and_header_only_files_rejected(self) -> None:
        empty_path = os.path.join(self.tmpdir, "empty.csv")
        with open(empty_path, "w", encoding="utf-8"):
            pass
        with self.assertRaises(fd.DataValidationError):
            fd.load_transactions(empty_path, require_label=False, require_time=False)

        blank_path = os.path.join(self.tmpdir, "blank.csv")
        with open(blank_path, "w", encoding="utf-8") as handle:
            handle.write("   \n")
        with self.assertRaises(fd.DataValidationError):
            fd.load_transactions(blank_path, require_label=False, require_time=False)

        header_only = self._write_csv("header_only.csv", _make_minimal_frame(rows=1).iloc[0:0])
        with self.assertRaises(fd.DataValidationError) as ctx:
            fd.load_transactions(header_only, require_label=True, require_time=True)
        self.assertIn("no transaction rows", str(ctx.exception))

    def test_malformed_input_publishes_no_artifacts(self) -> None:
        bad_df = _make_minimal_frame(rows=10).astype(str)
        bad_df.loc[3, "Amount"] = "bad"
        bad_path = self._write_csv("bad.csv", bad_df)

        run_dir = os.path.join(self.tmpdir, "nested", "should_not_exist")
        with self.assertRaises(fd.DataValidationError):
            fd.train(bad_path, run_dir)
        self.assertFalse(os.path.exists(run_dir))
        self.assertFalse(os.path.exists(os.path.dirname(run_dir)))

        valid_run = os.path.join(self.tmpdir, "valid_run")
        fd.train(LABELED_FIXTURE, valid_run)
        out_csv = os.path.join(self.tmpdir, "should_not_exist.csv")
        with self.assertRaises(fd.DataValidationError):
            fd.score(bad_path, os.path.join(valid_run, "model.joblib"), out_csv)
        self.assertFalse(os.path.exists(out_csv))

    def test_output_path_collisions_rejected(self) -> None:
        run_dir = os.path.join(self.tmpdir, "run")
        fd.train(LABELED_FIXTURE, run_dir)
        model_path = os.path.join(run_dir, "model.joblib")
        input_copy = os.path.join(self.tmpdir, "input.csv")
        shutil.copy2(UNLABELED_FIXTURE, input_copy)
        original_hash = fd.sha256_file(input_copy)

        # Direct collision with input
        with self.assertRaises(ValueError):
            fd.score(input_copy, model_path, input_copy)

        # Symlink collision with input
        link_path = os.path.join(self.tmpdir, "input_link.csv")
        os.symlink(input_copy, link_path)
        with self.assertRaises(ValueError):
            fd.score(input_copy, model_path, link_path)

        # Collision with model file
        with self.assertRaises(ValueError):
            fd.score(input_copy, model_path, model_path)

        # Collision with directory
        with self.assertRaises(ValueError):
            fd.score(input_copy, model_path, self.tmpdir)

        # Input file untouched
        self.assertEqual(fd.sha256_file(input_copy), original_hash)

    def test_row_numbering_and_audit_duplicates_and_conflicts(self) -> None:
        df = _make_minimal_frame(rows=6)
        exact_dup = df.iloc[[0]].copy()
        conflict_dup = df.iloc[[1]].copy()
        conflict_dup["Class"] = 1 - int(conflict_dup["Class"].iloc[0])
        combined = pd.concat([df, exact_dup, conflict_dup], ignore_index=True)
        path = self._write_csv("dupes.csv", combined)

        loaded = fd.load_transactions(path, require_label=True, require_time=True)
        np.testing.assert_array_equal(loaded["row_number"].to_numpy(), np.arange(1, 9))

        audit = fd.audit_dataset(loaded)
        self.assertEqual(audit["rows"], 8)
        self.assertEqual(audit["exact_duplicate_records_beyond_first"], 1)
        self.assertEqual(audit["records_in_exact_duplicate_groups"], 2)
        self.assertEqual(audit["raw_field_groups_with_conflicting_labels"], 1)
        self.assertEqual(audit["records_in_conflicting_label_groups"], 2)
        self.assertTrue(all(v == 0 for v in audit["missing_values"].values()))
        self.assertFalse(audit["publisher_comparison"]["rows_match"])


class TestChronologyAndLeakage(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="fraud-test-chrono-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_shuffled_repeated_timestamps_split_disjointly_and_deterministically(self) -> None:
        df = fd.load_transactions(LABELED_FIXTURE, require_label=True, require_time=True)
        # Fixture is shuffled; confirm raw Time is not already sorted.
        self.assertFalse(df["Time"].is_monotonic_increasing)

        parts1, manifest1 = fd.chronological_split(df)
        parts2, manifest2 = fd.chronological_split(df)
        self.assertEqual(manifest1, manifest2)

        # Strict chronological separation and disjoint timestamps.
        self.assertLess(parts1["train"]["Time"].max(), parts1["validation"]["Time"].min())
        self.assertLess(parts1["validation"]["Time"].max(), parts1["test"]["Time"].min())

        time_sets = {name: set(part["Time"].tolist()) for name, part in parts1.items()}
        self.assertEqual(time_sets["train"] & time_sets["validation"], set())
        self.assertEqual(time_sets["validation"] & time_sets["test"], set())
        self.assertEqual(time_sets["train"] & time_sets["test"], set())

        # Every input row appears in exactly one partition, sorted by (Time, row_number).
        all_rows = []
        for part in parts1.values():
            rows = part["row_number"].to_numpy()
            times = part["Time"].to_numpy()
            order = np.lexsort((rows, times))
            np.testing.assert_array_equal(order, np.arange(len(part)))
            all_rows.extend(rows.tolist())
        self.assertEqual(sorted(all_rows), list(range(1, len(df) + 1)))

    def test_all_identical_timestamps_rejected(self) -> None:
        df = _make_minimal_frame(rows=20)
        df["Time"] = 100.0
        df["row_number"] = np.arange(1, len(df) + 1)
        with self.assertRaises(fd.DataValidationError):
            fd.chronological_split(df)

    def test_fitted_features_exclude_label_time_and_row_number(self) -> None:
        run_dir = os.path.join(self.tmpdir, "run")
        record = fd.train(LABELED_FIXTURE, run_dir)
        bundle = fd.load_model(os.path.join(run_dir, "model.joblib"))

        fitted_names = [str(c) for c in bundle["estimator"].feature_names_in_]
        self.assertEqual(fitted_names, fd.FEATURES)
        self.assertEqual(bundle["features"], fd.FEATURES)
        self.assertEqual(record["model"]["features"], fd.FEATURES)
        for forbidden in ("Class", "Time", "row_number"):
            self.assertNotIn(forbidden, fitted_names)

    def test_anomaly_score_sign_convention_normal_cluster_plus_outlier(self) -> None:
        from sklearn.ensemble import IsolationForest

        rng = np.random.default_rng(123)
        normal = rng.normal(0.0, 0.25, size=(300, len(fd.FEATURES)))
        outlier = np.full((1, len(fd.FEATURES)), 25.0)
        frame = pd.DataFrame(np.vstack([normal, outlier]), columns=fd.FEATURES)

        est = IsolationForest(**fd.IFOREST_PARAMS)
        est.fit(frame)
        scores = fd.isolation_scores(est, frame)
        self.assertGreater(scores[-1], scores[:-1].max())


class TestCutoffsAndTies(unittest.TestCase):
    def test_plan_worked_example(self) -> None:
        # Plan §3 worked example: N=10,000, 0.5% budget -> K=50.
        # 47 rows score above a tied group of 8; 12 of the 47 are fraud, 8 more frauds missed (20 total).
        n = 10_000
        scores = np.concatenate([
            np.linspace(1.0, 0.9, 47),  # 47 rows above the tie group
            np.full(8, 0.8),            # tied group of 8 -> cumulative 55 > 50
            np.full(n - 55, 0.1),       # remaining 9,945 rows
        ])
        labels = np.zeros(n, dtype=int)
        labels[:12] = 1                 # 12 TP in top 47
        labels[47:55] = 1               # 8 FN in the excluded tie group

        choice = fd.select_cutoff(scores, "0.005")
        self.assertEqual(choice["K"], 50)
        self.assertTrue(choice["active"])
        self.assertEqual(choice["flagged"], 47)
        self.assertEqual(choice["cutoff"], 0.9)

        flags = fd.apply_cutoff(scores, choice["cutoff"])
        cm = fd.confusion_metrics(labels, flags)
        self.assertEqual((cm["TP"], cm["FP"], cm["FN"], cm["TN"]), (12, 35, 8, 9945))
        self.assertAlmostEqual(cm["precision"], 12 / 47, places=12)
        self.assertAlmostEqual(cm["recall"], 12 / 20, places=12)

    def test_exact_cutoff_match_and_budget_boundary(self) -> None:
        # 10 rows, rate 0.3 -> K = 3. Scores: [5, 4, 4, 3, ...] -> cumulative [1, 3, 4, ...]
        scores = np.array([5.0, 4.0, 4.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0])
        choice = fd.select_cutoff(scores, "0.3")
        self.assertEqual(choice["K"], 3)
        self.assertTrue(choice["active"])
        self.assertEqual(choice["cutoff"], 4.0)
        self.assertEqual(choice["flagged"], 3)
        flags = fd.apply_cutoff(scores, choice["cutoff"])
        np.testing.assert_array_equal(flags, np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]))

        # Exact cutoff equality on new scores
        new_scores = np.array([4.0, 3.9999999999999, 4.0000000000001])
        np.testing.assert_array_equal(fd.apply_cutoff(new_scores, 4.0), np.array([1, 0, 1]))

    def test_empty_queue_when_k_zero_or_top_group_exceeds_k(self) -> None:
        small = np.array([0.9, 0.5, 0.1])
        choice_zero = fd.select_cutoff(small, "0.005")
        self.assertEqual(choice_zero["K"], 0)
        self.assertFalse(choice_zero["active"])
        self.assertIsNone(choice_zero["cutoff"])
        self.assertEqual(choice_zero["flagged"], 0)
        np.testing.assert_array_equal(fd.apply_cutoff(small, choice_zero["cutoff"]), np.zeros(3, dtype=np.int64))

        tied_top = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        choice_tie = fd.select_cutoff(tied_top, "0.3")  # K = 3, top tie group has 4
        self.assertEqual(choice_tie["K"], 3)
        self.assertFalse(choice_tie["active"])
        self.assertIsNone(choice_tie["cutoff"])
        self.assertEqual(choice_tie["flagged"], 0)
        np.testing.assert_array_equal(fd.apply_cutoff(tied_top, choice_tie["cutoff"]), np.zeros(10, dtype=np.int64))

    def test_ranking_tie_order_uses_row_number_ascending(self) -> None:
        scores = np.array([0.5, 0.9, 0.5, 0.9, 0.5])
        rows = np.array([30, 20, 10, 5, 40])
        order = fd.ranking_order(scores, rows)
        np.testing.assert_array_equal(rows[order], np.array([5, 20, 10, 30, 40]))
        np.testing.assert_array_equal(scores[order], np.array([0.9, 0.9, 0.5, 0.5, 0.5]))

    def test_amount_only_baseline_uses_same_budgets_and_tie_rule(self) -> None:
        # 200 rows -> 0.5% budget gives K=1, 1.0% gives K=2, 0.1% gives K=0.
        amounts = np.full(200, 10.0)
        amounts[0] = 500.0
        amounts[1] = 250.0
        amounts[2] = 250.0
        df = pd.DataFrame({"Amount": amounts})
        scores = fd.amount_scores(df)
        c_01 = fd.select_cutoff(scores, fd.BUDGETS["0.1%"])
        c_05 = fd.select_cutoff(scores, fd.BUDGETS["0.5%"])
        c_10 = fd.select_cutoff(scores, fd.BUDGETS["1.0%"])
        self.assertFalse(c_01["active"])
        self.assertIsNone(c_01["cutoff"])
        self.assertEqual((c_05["active"], c_05["cutoff"], c_05["flagged"]), (True, 500.0, 1))
        # At K=2, the tied pair at 250.0 would bring cumulative to 3 > 2, so cutoff stays at 500.0.
        self.assertEqual((c_10["active"], c_10["cutoff"], c_10["flagged"]), (True, 500.0, 1))


class TestMetricsAndDegenerateCases(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="fraud-test-metrics-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_hand_calculated_metrics_all_four_outcomes(self) -> None:
        # 10 rows sorted by score:
        # scores: [0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
        # labels: [   1,    0,    1,    0,    0,    1,    0,    0,    0,    0] -> 3 positives
        # flags:  [   1,    1,    1,    1,    0,    0,    0,    0,    0,    0] -> 4 alerts (cutoff 0.70)
        # TP=2, FP=2, FN=1, TN=5
        scores = np.array([0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10])
        labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        rows = np.arange(1, 11)
        flags = fd.apply_cutoff(scores, 0.70)

        cm = fd.confusion_metrics(labels, flags)
        self.assertEqual((cm["TP"], cm["FP"], cm["FN"], cm["TN"]), (2, 2, 1, 5))
        self.assertEqual(cm["total"], 10)
        self.assertEqual(cm["positive_support"], 3)
        self.assertEqual(cm["alerts"], 4)
        self.assertAlmostEqual(cm["precision"], 2 / 4)
        self.assertAlmostEqual(cm["recall"], 2 / 3)
        self.assertAlmostEqual(cm["alert_rate"], 0.4)
        self.assertAlmostEqual(cm["alerts_per_100k"], 40_000.0)

        # Top-K at rate 0.3 -> K = 3 -> rows [1, 2, 3] -> labels [1, 0, 1] -> TP=2
        tk = fd.top_k_metrics(labels, scores, rows, "0.3")
        self.assertEqual(tk["K"], 3)
        self.assertEqual(tk["TP_in_top_K"], 2)
        self.assertAlmostEqual(tk["precision_at_K"], 2 / 3)
        self.assertAlmostEqual(tk["recall_at_K"], 2 / 3)

        # Hand-calculated AP: positives at ranks 1, 3, 6 -> P@1=1/1, P@3=2/3, P@6=3/6=1/2.
        # Each recall increment is 1/3 -> AP = (1/3)*(1 + 2/3 + 1/2) = 13/18.
        ap = fd.average_precision(labels, scores)
        self.assertTrue(ap["available"])
        self.assertAlmostEqual(ap["AP"], 13 / 18, places=12)

    def test_no_alert_and_no_fraud_degenerate_cases(self) -> None:
        labels = np.array([1, 0, 0, 1, 0])
        zero_flags = np.zeros(5, dtype=np.int64)
        no_alert = fd.confusion_metrics(labels, zero_flags)
        self.assertIsNone(no_alert["precision"])
        self.assertEqual(fd.fmt_rate(no_alert["precision"]), "N/A")
        self.assertEqual(no_alert["recall"], 0.0)
        self.assertEqual(no_alert["alerts"], 0)
        self.assertIn('"precision": null', json.dumps(no_alert))

        no_fraud_labels = np.zeros(5, dtype=np.int64)
        scores = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        flags = np.array([1, 0, 0, 0, 0])
        no_fraud = fd.confusion_metrics(no_fraud_labels, flags)
        self.assertEqual(no_fraud["precision"], 0.0)
        self.assertIsNone(no_fraud["recall"])
        self.assertEqual(fd.fmt_rate(no_fraud["recall"]), "N/A")

        ap = fd.average_precision(no_fraud_labels, scores)
        self.assertFalse(ap["available"])
        self.assertIsNone(ap["AP"])
        self.assertEqual(fd.fmt_number(ap["AP"]), "N/A")

        tk = fd.top_k_metrics(no_fraud_labels, scores, np.arange(1, 6), "0.4")
        self.assertEqual(tk["K"], 2)
        self.assertEqual(tk["precision_at_K"], 0.0)
        self.assertIsNone(tk["recall_at_K"])

    def test_single_class_validation_and_test_end_to_end(self) -> None:
        df = pd.read_csv(LABELED_FIXTURE).sort_values("Time").reset_index(drop=True)
        # Keep frauds only in the first 50% so validation and test are strictly single-class (Class=0).
        df.loc[len(df) // 2 :, "Class"] = 0
        csv_path = os.path.join(self.tmpdir, "single_class_val_test.csv")
        df.to_csv(csv_path, index=False)

        run_dir = os.path.join(self.tmpdir, "run")
        run_record = fd.train(csv_path, run_dir)
        self.assertTrue(run_record["validation"]["single_class"])
        self.assertFalse(run_record["validation"]["metrics"]["isolation_forest"]["average_precision"]["available"])
        self.assertIsNone(run_record["validation"]["metrics"]["isolation_forest"]["average_precision"]["AP"])
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "development_overview.png")))
        self.assertFalse(os.path.exists(os.path.join(run_dir, "validation_precision_recall.png")))

        metrics = fd.evaluate(csv_path, run_dir)
        self.assertTrue(metrics["test"]["single_class"])
        self.assertFalse(metrics["test"]["ranking_comparison_available"])
        self.assertIsNone(metrics["methods"]["isolation_forest"]["average_precision"]["AP"])
        self.assertIsNone(metrics["methods"]["isolation_forest"]["budgets"]["0.5%"]["at_cutoff"]["recall"])
        self.assertFalse(os.path.exists(os.path.join(run_dir, "precision_recall.png")))


class TestPersistenceEvaluationAndScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="fraud-test-e2e-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_serialization_parity_evaluation_guards_and_scoring_roundtrip(self) -> None:
        run_dir = os.path.join(self.tmpdir, "run")
        run_record = fd.train(LABELED_FIXTURE, run_dir)

        # Artifacts created
        model_path = os.path.join(run_dir, "model.joblib")
        self.assertTrue(os.path.isfile(model_path))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "run.json")))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "development_overview.png")))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "validation_precision_recall.png")))

        # Training refuses to overwrite existing run_dir
        with self.assertRaises(FileExistsError):
            fd.train(LABELED_FIXTURE, run_dir)

        # Compare pre-serialization in-memory refit vs serialized model.joblib
        bundle = fd.load_model(model_path)
        unlabeled_df = fd.load_transactions(UNLABELED_FIXTURE, require_label=False, require_time=False)
        from sklearn.ensemble import IsolationForest
        labeled_df = fd.load_transactions(LABELED_FIXTURE, require_label=True, require_time=True)
        parts, _ = fd.chronological_split(labeled_df)
        fresh_est = IsolationForest(**fd.IFOREST_PARAMS).fit(fd.feature_matrix(parts["train"]))
        pre_scores = fd.isolation_scores(fresh_est, unlabeled_df)
        post_scores = fd.isolation_scores(bundle["estimator"], unlabeled_df, bundle["features"])
        cutoff = bundle["cutoffs"]["isolation_forest"][bundle["main_budget"]]["cutoff"]

        np.testing.assert_allclose(pre_scores, post_scores, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(fd.apply_cutoff(pre_scores, cutoff), fd.apply_cutoff(post_scores, cutoff))

        # Evaluate refuses mismatched dataset
        altered_path = os.path.join(self.tmpdir, "altered.csv")
        altered = pd.read_csv(LABELED_FIXTURE)
        altered.loc[0, "Amount"] = float(altered.loc[0, "Amount"]) + 1.0
        altered.to_csv(altered_path, index=False)
        with self.assertRaises(fd.DataValidationError):
            fd.evaluate(altered_path, run_dir)

        # Evaluate succeeds once and refuses a second run
        metrics = fd.evaluate(LABELED_FIXTURE, run_dir)
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "metrics.json")))
        self.assertTrue(os.path.isfile(os.path.join(run_dir, "precision_recall.png")))
        self.assertEqual(metrics["dataset_sha256"], run_record["dataset"]["sha256"])
        with self.assertRaises(FileExistsError):
            fd.evaluate(LABELED_FIXTURE, run_dir)

        # Score unlabeled CSV without Class, and also without Time
        queue_path = os.path.join(self.tmpdir, "review_queue.csv")
        summary = fd.score(UNLABELED_FIXTURE, model_path, queue_path)
        queue = pd.read_csv(queue_path, float_precision="round_trip")
        self.assertEqual(list(queue.columns), ["row_number", "anomaly_score", "review_flag"])
        self.assertEqual(len(queue), len(unlabeled_df))
        self.assertEqual(summary["rows"], len(unlabeled_df))

        # Verify sorting (score desc, row_number asc) and joinability by row_number
        expected_order = fd.ranking_order(queue["anomaly_score"].to_numpy(), queue["row_number"].to_numpy())
        np.testing.assert_array_equal(expected_order, np.arange(len(queue)))
        joined = queue.merge(unlabeled_df[["row_number", "Amount"]], on="row_number", how="inner")
        self.assertEqual(len(joined), len(unlabeled_df))

        # Re-evaluating saved cutoff on reloaded CSV scores reproduces exact review_flag
        reproduced_flags = fd.apply_cutoff(queue["anomaly_score"].to_numpy(), cutoff)
        np.testing.assert_array_equal(queue["review_flag"].to_numpy(), reproduced_flags)

        # Scoring works when Time is omitted as well
        no_time_path = os.path.join(self.tmpdir, "no_time.csv")
        pd.read_csv(UNLABELED_FIXTURE).drop(columns=["Time"]).to_csv(no_time_path, index=False)
        no_time_queue = os.path.join(self.tmpdir, "no_time_queue.csv")
        fd.score(no_time_path, model_path, no_time_queue)
        q2 = pd.read_csv(no_time_queue, float_precision="round_trip")
        np.testing.assert_array_equal(queue["row_number"].to_numpy(), q2["row_number"].to_numpy())
        np.testing.assert_allclose(queue["anomaly_score"].to_numpy(), q2["anomaly_score"].to_numpy(), rtol=0, atol=1e-15)
        np.testing.assert_array_equal(queue["review_flag"].to_numpy(), q2["review_flag"].to_numpy())

    def test_score_exact_equality_survives_csv_roundtrip(self) -> None:
        run_dir = os.path.join(self.tmpdir, "run")
        fd.train(LABELED_FIXTURE, run_dir)
        model_path = os.path.join(run_dir, "model.joblib")
        bundle = fd.load_model(model_path)

        unlabeled_df = fd.load_transactions(UNLABELED_FIXTURE, require_label=False, require_time=False)
        scores = fd.isolation_scores(bundle["estimator"], unlabeled_df, bundle["features"])
        # Pin cutoff to the exact float64 score of row index 17.
        exact_cutoff = float(scores[17])
        bundle["cutoffs"]["isolation_forest"][bundle["main_budget"]]["cutoff"] = exact_cutoff
        patched_model = os.path.join(self.tmpdir, "patched.joblib")
        joblib.dump(bundle, patched_model)

        queue_path = os.path.join(self.tmpdir, "exact_queue.csv")
        fd.score(UNLABELED_FIXTURE, patched_model, queue_path)
        queue = pd.read_csv(queue_path, float_precision="round_trip")

        # The exact-match row must be flagged and stay == cutoff after CSV float round-trip.
        target_row = queue.loc[queue["row_number"] == 18]
        self.assertEqual(len(target_row), 1)
        self.assertEqual(int(target_row["review_flag"].iloc[0]), 1)
        self.assertEqual(float(target_row["anomaly_score"].iloc[0]), exact_cutoff)
        np.testing.assert_array_equal(
            queue["review_flag"].to_numpy(),
            fd.apply_cutoff(queue["anomaly_score"].to_numpy(), exact_cutoff),
        )

    def test_cli_exit_codes(self) -> None:
        run_dir = os.path.join(self.tmpdir, "cli_run")
        queue_path = os.path.join(self.tmpdir, "cli_queue.csv")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(fd.main(["train", "--input", LABELED_FIXTURE, "--run-dir", run_dir]), 0)
            self.assertEqual(fd.main(["evaluate", "--input", LABELED_FIXTURE, "--run-dir", run_dir]), 0)
            self.assertEqual(
                fd.main([
                    "score",
                    "--input", UNLABELED_FIXTURE,
                    "--model", os.path.join(run_dir, "model.joblib"),
                    "--output", queue_path,
                ]),
                0,
            )

        err_buf = io.StringIO()
        with contextlib.redirect_stderr(err_buf):
            self.assertEqual(
                fd.main(["train", "--input", os.path.join(self.tmpdir, "missing.csv"), "--run-dir", run_dir + "_2"]),
                2,
            )
            self.assertEqual(
                fd.main([
                    "score",
                    "--input", UNLABELED_FIXTURE,
                    "--model", os.path.join(run_dir, "model.joblib"),
                    "--output", UNLABELED_FIXTURE,
                ]),
                2,
            )
        self.assertIn("error:", err_buf.getvalue())


class TestPromotedModels(unittest.TestCase):
    def test_fixed_cutoff_scoring_roundtrip(self) -> None:
        from sklearn.ensemble import IsolationForest, RandomForestClassifier

        source = pd.read_csv(LABELED_FIXTURE, float_precision="round_trip")
        with tempfile.TemporaryDirectory() as directory:
            for kind, score_column in (("random_forest", "fraud_score"), ("if_combined", "anomaly_score")):
                with self.subTest(model=kind):
                    tuned = kind == "if_combined"
                    features = fd.FEATURES[:-1] if tuned else fd.FEATURES
                    history = os.path.join(directory, f"{kind}_history.csv")
                    source[["Time", *features] + ([] if tuned else ["Class"])].to_csv(history, index=False)
                    batch = os.path.join(directory, f"{kind}_batch.csv")
                    source[features[::-1]].head(100).to_csv(batch, index=False)
                    run_dir = os.path.join(directory, kind)
                    queue_path = os.path.join(directory, f"{kind}_queue.csv")
                    record = fd.train(history, run_dir, kind)
                    self.assertEqual(record["validation"]["metrics"] is None, tuned)
                    model_path = os.path.join(run_dir, "model.joblib")
                    bundle = fd.load_model(model_path)
                    self.assertEqual(bundle["format_version"], 2)
                    self.assertEqual(bundle["features"], features)

                    training = fd.load_transactions(history, require_label=not tuned, require_time=True, features=features)
                    parts, _ = fd.chronological_split(training)
                    fit = parts["train"]
                    incoming = fd.load_transactions(batch, require_label=False, require_time=False, features=features)
                    if tuned:
                        reference = IsolationForest(**fd.IF_COMBINED_PARAMS).fit(fit[features])
                        scores = -reference.score_samples(incoming[features])
                        validation_scores = -reference.score_samples(parts["validation"][features])
                    else:
                        reference = RandomForestClassifier(**fd.RANDOM_FOREST_PARAMS).fit(fit[features], fit["Class"])
                        reference.set_params(n_jobs=1)
                        scores = reference.predict_proba(incoming[features])[:, 1]
                        validation_scores = reference.predict_proba(parts["validation"][features])[:, 1]

                    summary = fd.score(batch, model_path, queue_path)
                    queue = pd.read_csv(queue_path, float_precision="round_trip")
                    self.assertEqual(list(queue.columns), ["row_number", score_column, "review_flag"])
                    expected_order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                    np.testing.assert_array_equal(queue["row_number"], np.asarray(expected_order) + 1)
                    np.testing.assert_array_equal(queue[score_column], scores[expected_order])
                    cutoff = record["validation"]["cutoffs"][kind]["0.5%"]["cutoff"]
                    self.assertEqual(cutoff, fd.select_cutoff(validation_scores, "0.005")["cutoff"])
                    self.assertEqual(summary["cutoff"], cutoff)
                    self.assertEqual(list(queue["review_flag"]),
                                     [int(cutoff is not None and scores[i] >= cutoff) for i in expected_order])

                    missing = os.path.join(directory, f"{kind}_missing.csv")
                    source[features[1:]].head(5).to_csv(missing, index=False)
                    missing_queue = os.path.join(directory, f"{kind}_missing_queue.csv")
                    with self.assertRaisesRegex(fd.DataValidationError, "missing required column.*V1"):
                        fd.score(missing, model_path, missing_queue)
                    self.assertFalse(os.path.exists(missing_queue))
                    if tuned:
                        with self.assertRaisesRegex(fd.DataValidationError, "missing required column.*Class"):
                            fd.evaluate(history, run_dir)
                    else:
                        metrics = fd.evaluate(history, run_dir)
                        self.assertEqual(list(metrics["methods"]), [kind])
                        self.assertEqual(metrics["methods"][kind]["budgets"]["0.5%"]["frozen_cutoff"], cutoff)

    def test_random_forest_rejects_one_class_in_fit_partition(self) -> None:
        history = _make_minimal_frame()
        history.loc[history["Time"] < 18, "Class"] = 0
        with tempfile.TemporaryDirectory() as directory:
            input_path = os.path.join(directory, "history.csv")
            run_dir = os.path.join(directory, "run")
            history.to_csv(input_path, index=False)
            with self.assertRaisesRegex(fd.DataValidationError, "training rows must contain both"):
                fd.train(input_path, run_dir, "random_forest")
            self.assertFalse(os.path.exists(run_dir))


if __name__ == "__main__":
    unittest.main()
