"""Unit and integration tests for experiments.py (plan.md section 8)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import experiments as ex
import fraud_detection as fd

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_FIXTURE = os.path.join(TESTS_DIR, "fixtures", "labeled_small.csv")


class TestDevelopmentFolds(unittest.TestCase):
    def test_expanding_folds_respect_chronology_and_timestamp_groups(self) -> None:
        df = fd.load_transactions(LABELED_FIXTURE, require_label=True, require_time=True)
        parts, _ = fd.chronological_split(df)
        folds, manifest = ex.development_folds(parts)

        self.assertEqual(len(folds), 3)
        self.assertEqual(len(manifest["blocks"]), 4)
        self.assertEqual(len(manifest["folds"]), 3)

        # Blocks strictly increasing in Time with no shared timestamps.
        block_times = [b["time_range_seconds"] for b in manifest["blocks"]]
        for prev, nxt in zip(block_times[:-1], block_times[1:]):
            self.assertLess(prev[1], nxt[0])

        # Every fold fit precedes eval in time and grows monotonically.
        prev_fit_len = 0
        for fold in folds:
            self.assertGreater(len(fold["fit"]), prev_fit_len)
            prev_fit_len = len(fold["fit"])
            self.assertLess(fold["fit"][fd.TIME].max(), fold["eval"][fd.TIME].min())
            self.assertEqual(
                set(fold["fit"][fd.TIME].tolist()) & set(fold["eval"][fd.TIME].tolist()),
                set(),
            )

        # Last fold matches train -> validation exactly.
        np.testing.assert_array_equal(
            folds[-1]["fit"][fd.ROW].to_numpy(),
            parts["train"][fd.ROW].to_numpy(),
        )
        np.testing.assert_array_equal(
            folds[-1]["eval"][fd.ROW].to_numpy(),
            parts["validation"][fd.ROW].to_numpy(),
        )

    def test_single_timestamp_train_rejected(self) -> None:
        df = fd.load_transactions(LABELED_FIXTURE, require_label=True, require_time=True)
        parts, _ = fd.chronological_split(df)
        parts["train"] = parts["train"].copy()
        parts["train"][fd.TIME] = 1.0
        with self.assertRaises(fd.DataValidationError):
            ex.development_folds(parts)


class TestCandidatesAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = fd.load_transactions(LABELED_FIXTURE, require_label=True, require_time=True)
        cls.parts, _ = fd.chronological_split(cls.df)

    def test_log1p_amount_preserves_input_and_columns(self) -> None:
        X = self.parts["train"].loc[:10, fd.FEATURES].copy()
        orig_amount = X["Amount"].to_numpy().copy()
        transformed = ex.log1p_amount(X)
        self.assertEqual(list(transformed.columns), list(fd.FEATURES))
        np.testing.assert_array_equal(X["Amount"].to_numpy(), orig_amount)
        np.testing.assert_allclose(transformed["Amount"].to_numpy(), np.log1p(orig_amount))

    def test_scalers_fit_on_fit_rows_only(self) -> None:
        # Scoring a fixed probe row must be invariant to other rows in the eval batch.
        probe = self.parts["validation"].iloc[[0]].copy()
        extreme = self.parts["validation"].iloc[1:20].copy()
        extreme["Amount"] = 1e9
        extreme["V1"] = 1e6
        batch_a = pd.concat([probe, self.parts["validation"].iloc[1:20]], ignore_index=True)
        batch_b = pd.concat([probe, extreme], ignore_index=True)

        for cand in (
            ex.make_iforest("if_log1p_std", "desc", preprocess="log1p_standard"),
            ex.make_iforest("if_log1p_rob", "desc", preprocess="log1p_robust"),
            ex.make_supervised("logistic_regression"),
            ex.make_lof("lof_std", "standard", n_neighbors=10, max_fit_rows=300),
            ex.make_autoencoder("ae_test", seed=0, preprocess="log1p_standard"),
        ):
            scorer, _ = cand.builder(self.parts["train"])
            s_a = scorer(batch_a)[0]
            s_b = scorer(batch_b)[0]
            self.assertAlmostEqual(float(s_a), float(s_b), places=12, msg=f"leakage in {cand.name}")

    def test_autoencoder_trains_strictly_on_class_zero(self) -> None:
        train_a = self.parts["train"].copy()
        train_b = self.parts["train"].copy()
        fraud_mask = train_b[fd.LABEL] == 1
        self.assertGreater(int(fraud_mask.sum()), 0)
        # Corrupt all Class=1 feature rows heavily; autoencoder must ignore them completely.
        train_b.loc[fraud_mask, fd.FEATURES] = 9999.0

        ae = ex.make_autoencoder("ae_s0", seed=0, preprocess="log1p_standard")
        scorer_a, info_a = ae.builder(train_a)
        scorer_b, info_b = ae.builder(train_b)
        self.assertEqual(info_a["fit_rows"], int((train_a[fd.LABEL] == 0).sum()))
        self.assertEqual(info_a["fit_rows_excluded_class_1"], int(fraud_mask.sum()))
        self.assertEqual(info_a["fit_rows"], info_b["fit_rows"])
        np.testing.assert_allclose(
            scorer_a(self.parts["validation"]),
            scorer_b(self.parts["validation"]),
            rtol=0,
            atol=1e-12,
        )

    def test_single_class_fit_is_skipped_gracefully(self) -> None:
        folds, _ = ex.development_folds(self.parts)
        folds_mod = []
        for i, fold in enumerate(folds):
            fit_copy = fold["fit"].copy()
            if i == 0:
                fit_copy[fd.LABEL] = 0
            folds_mod.append({"name": fold["name"], "fit": fit_copy, "eval": fold["eval"]})

        cand = ex.make_supervised("logistic_regression")
        res = ex.run_candidate(cand, folds_mod, self.parts["test"], log=lambda *_: None)
        self.assertIn("skipped", res["folds"]["fold1"])
        self.assertNotIn("skipped", res["folds"]["fold2"])
        self.assertIsNotNone(res["validation_scores"])
        summary = ex.cv_summary(res["folds"])
        self.assertEqual(summary["folds_used"], 2)
        self.assertEqual(summary["folds_skipped"], 1)

    def test_score_direction_for_unsupervised_and_autoencoder(self) -> None:
        rng = np.random.default_rng(7)
        n = 400
        normal = rng.normal(0.0, 0.25, size=(n, len(fd.FEATURES)))
        normal[:, -1] = np.abs(normal[:, -1]) * 10.0  # nonnegative Amount
        outlier = np.full((1, len(fd.FEATURES)), 15.0)

        fit_df = pd.DataFrame(normal, columns=fd.FEATURES)
        fit_df[fd.TIME] = np.arange(n, dtype=float)
        fit_df[fd.LABEL] = 0
        fit_df[fd.ROW] = np.arange(1, n + 1)

        eval_arr = np.vstack([normal[:50], outlier])
        eval_df = pd.DataFrame(eval_arr, columns=fd.FEATURES)
        eval_df[fd.TIME] = np.arange(n, n + len(eval_df), dtype=float)
        eval_df[fd.LABEL] = 0
        eval_df[fd.ROW] = np.arange(1, len(eval_df) + 1)

        for cand in (
            ex.make_iforest("if_rob", "desc", preprocess="log1p_robust"),
            ex.make_lof("lof_k20", "standard", n_neighbors=20),
            ex.make_autoencoder("ae_s0", seed=0, preprocess="log1p_standard"),
        ):
            scorer, _ = cand.builder(fit_df)
            scores = scorer(eval_df)
            self.assertGreater(scores[-1], scores[:-1].max(), msg=f"bad score sign in {cand.name}")


class TestQueueStabilization(unittest.TestCase):
    def test_batch_top_k_flags_and_tie_order(self) -> None:
        scores = np.array([0.5, 0.9, 0.9, 0.1, 0.4, 0.4, 0.4])
        rows = np.array([10, 20, 5, 30, 40, 12, 25])
        batches = np.array([0, 0, 0, 0, 1, 1, 1])

        # K=1 per batch: batch 0 picks score 0.9 with smaller row_number (5 -> idx 2);
        # batch 1 picks score 0.4 with smallest row_number (12 -> idx 5).
        flags = ex.batch_top_k_flags(scores, rows, batches, lambda n: 1)
        np.testing.assert_array_equal(flags, np.array([0, 0, 1, 0, 0, 1, 0]))

        # K=0 flags nothing; K > batch size flags entire batch.
        np.testing.assert_array_equal(
            ex.batch_top_k_flags(scores, rows, batches, lambda n: 0),
            np.zeros(len(scores), dtype=np.int64),
        )
        np.testing.assert_array_equal(
            ex.batch_top_k_flags(scores, rows, batches, lambda n: 99),
            np.ones(len(scores), dtype=np.int64),
        )

    def test_rolling_cutoffs_causal_no_current_or_future_leakage(self) -> None:
        ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        apply_a = np.array([10.0, 0.5, 6.0, 2.5])
        q_a = ex.rolling_cutoffs(ref, apply_a, rate="0.2", window=5)

        # Row 0 cutoff uses ref [1,2,3,4,5] -> 80th percentile = 4.2
        self.assertAlmostEqual(float(q_a[0]), float(np.quantile(ref, 0.8)), places=12)
        # Row 1 cutoff uses [2,3,4,5,10] -> 80th percentile
        self.assertAlmostEqual(
            float(q_a[1]), float(np.quantile([2.0, 3.0, 4.0, 5.0, 10.0], 0.8)), places=12
        )

        # Mutating apply[2:] must not affect cutoffs for rows 0, 1, or 2.
        apply_b = apply_a.copy()
        apply_b[2] = 9999.0
        apply_b[3] = -9999.0
        q_b = ex.rolling_cutoffs(ref, apply_b, rate="0.2", window=5)
        np.testing.assert_allclose(q_a[:3], q_b[:3], rtol=0, atol=1e-15)

        # Window larger than reference rejected.
        with self.assertRaises(ValueError):
            ex.rolling_cutoffs(ref, apply_a, rate="0.2", window=10)

    def test_stability_metrics_hand_calculation(self) -> None:
        # 2 hours: hour 0 has 100 rows (1 alert = 1.0%, 1 TP), hour 1 has 100 rows (0 alerts = 0.0%, 1 FN)
        labels = np.zeros(200, dtype=int)
        labels[0] = 1
        labels[100] = 1
        flags = np.zeros(200, dtype=int)
        flags[0] = 1
        batches = np.array([0] * 100 + [1] * 100)

        m = ex.stability_metrics(labels, flags, batches, rate="0.005")
        self.assertEqual(m["target_alerts"], 1)
        self.assertEqual(m["alerts"], 1)
        self.assertAlmostEqual(m["alerts_vs_target"], 1.0)
        self.assertEqual(m["TP"], 1)
        self.assertEqual(m["frauds"], 2)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertEqual(m["hours"], 2)
        self.assertAlmostEqual(m["hourly_alert_rate_std"], 0.005)
        self.assertAlmostEqual(m["hourly_alert_rate_weighted_mean_abs_error"], 0.005)
        self.assertAlmostEqual(m["hourly_alert_rate_max_abs_error"], 0.005)


class TestTracksEndToEndAndCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="fraud-exp-test-")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_five_tracks_and_guards(self) -> None:
        quiet = lambda *_: None

        # Track 1
        d1 = os.path.join(self.tmpdir, "t1")
        r1 = ex.track1(LABELED_FIXTURE, d1, log=quiet)
        self.assertTrue(os.path.isfile(os.path.join(d1, "run.json")))
        self.assertTrue(os.path.isfile(os.path.join(d1, "metrics.json")))
        self.assertTrue(os.path.isfile(os.path.join(d1, "validation_precision_recall.png")))
        self.assertTrue(os.path.isfile(os.path.join(d1, "test_precision_recall.png")))
        self.assertTrue(r1["metrics"]["selection"]["selected_before_test_metrics"])
        best_sup = r1["metrics"]["selection"]["best_supervised_stage1"]
        self.assertIn(best_sup, ex.SUPERVISED_KINDS)
        self.assertIn("calibration", r1["metrics"]["test_exploratory"][best_sup])
        self.assertIn("calibration", r1["metrics"]["test_exploratory"][f"{best_sup}+if_feature"])
        self.assertIn(f"{best_sup}+rank_blend", r1["metrics"]["test_exploratory"])

        # Overwrite protection
        with self.assertRaises(FileExistsError):
            ex.track1(LABELED_FIXTURE, d1, log=quiet)

        # Track 2
        d2 = os.path.join(self.tmpdir, "t2")
        r2 = ex.track2(LABELED_FIXTURE, d2, log=quiet)
        self.assertTrue(r2["metrics"]["selection"]["selected_before_test_metrics"])
        self.assertIn("if_log1p_robust", r2["metrics"]["cv_summary"])
        self.assertIn("if_drop_amount", r2["metrics"]["cv_summary"])
        self.assertIn("top_tail_profile", r2["metrics"]["test_exploratory"]["if_log1p_robust"])

        # Track 3
        d3 = os.path.join(self.tmpdir, "t3")
        r3 = ex.track3(LABELED_FIXTURE, d3, budget_seconds=60.0, log=quiet)
        self.assertIn("lof_pilot", r3["metrics"])
        self.assertIn("lof_standard_k20", r3["metrics"]["cv_summary"])
        self.assertIn("peak_traced_memory_mb", r3["metrics"]["cv_folds"]["lof_standard_k20"]["fold1"])

        # Track 4
        d4 = os.path.join(self.tmpdir, "t4")
        r4 = ex.track4(LABELED_FIXTURE, d4, seeds=(0, 1), log=quiet)
        self.assertIn("seed_sensitivity", r4["metrics"])
        self.assertEqual(r4["metrics"]["seed_sensitivity"]["seeds"], [0, 1])
        self.assertIn("autoencoder_standard_seed0", r4["metrics"]["cv_summary"])

        # Track 5
        d5 = os.path.join(self.tmpdir, "t5")
        queue_models = (ex.REFERENCE, "logistic_regression", "if_combined")
        r5 = ex.track5(
            LABELED_FIXTURE,
            d5,
            models=queue_models,
            window=100,
            log=quiet,
        )
        self.assertIn("results", r5["metrics"])
        for model in queue_models:
            for win in ("dev_window1", "dev_window2", "test"):
                for b in fd.BUDGETS:
                    rules = r5["metrics"]["results"][model][win]["budgets"][b]
                    self.assertEqual(
                        set(rules),
                        {"static_cutoff", "batch_top_k_proportional", "batch_top_k_fixed_capacity", "rolling_quantile"},
                    )
        # Valid JSON serialized cleanly
        with open(os.path.join(d5, "metrics.json"), encoding="utf-8") as handle:
            parsed = json.load(handle)
        self.assertEqual(parsed["run_id"], r5["run_id"])

    def test_cli_exit_codes(self) -> None:
        d5 = os.path.join(self.tmpdir, "cli_t5")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ex.main(["track5", "--input", LABELED_FIXTURE, "--run-dir", d5, "--window", "100"])
        self.assertEqual(code, 0)
        with contextlib.redirect_stderr(err):
            # Existing run dir rejected
            self.assertEqual(
                ex.main(["track5", "--input", LABELED_FIXTURE, "--run-dir", d5, "--window", "100"]),
                2,
            )
            # Unknown model rejected
            self.assertEqual(
                ex.main(["track5", "--input", LABELED_FIXTURE, "--run-dir", d5 + "_2", "--models", "bogus", "--window", "100"]),
                2,
            )
        self.assertIn("error:", err.getvalue())


if __name__ == "__main__":
    unittest.main()
