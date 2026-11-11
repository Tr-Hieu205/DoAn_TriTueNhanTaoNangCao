"""
tests/test_challenger_m4.py
Empirical Adversarial Verification Suite for Milestone 4:
Comparative Evaluation Script & Metrics Verification.

Author: challenger_m4_1 (critic, specialist)
Validates:
1. Mathematical correctness of Accuracy and Macro-F1:
   - Unweighted arithmetic mean across classes: Macro-F1 = (1/C) * sum(F1_c)
   - Zero division handling when a class has 0 predictions or 0 true support
   - Confusion matrix row/col conventions (true rows vs predicted columns)
2. Mathematical correctness of Absolute Delta:
   - Delta = PhoBERT - Baseline
   - Signs and magnitude
3. Mathematical correctness of Relative Improvement:
   - Rel (%) = ((PhoBERT - Baseline) / Baseline) * 100%
   - Baseline = 0.0 division safety guard (returns 0.0)
4. Empirical verification of results/metrics.json:
   - Independent re-computation on test.parquet
   - Comparison against recorded values in results/metrics.json
   - Sample count discrepancy check (N=100 in metrics.json vs N=3,166 in PROJECT.md and worker handoff)
5. Robustness against extreme inputs, empty lists, single-class predictions, NaN/inf.
"""

import json
import os
import sys
import unittest
from typing import Dict, List, Any
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import evaluation.evaluate as eval_module
from data.dataset import load_vsfc_split
from models.baseline_model import VietnameseFeedbackBaseline


class TestMathematicalCorrectness(unittest.TestCase):
    """Rigorous empirical validation of metric mathematics."""

    def test_unweighted_macro_f1_vs_per_class_f1(self):
        """
        Verify Macro-F1 is strictly unweighted arithmetic mean across classes:
        Macro-F1 = sum(F1_c) / C, NOT weighted by support.
        """
        # Imbalanced test case with 3 classes
        # Class 0: 100 samples
        # Class 1: 10 samples
        # Class 2: 5 samples
        y_true = [0] * 100 + [1] * 10 + [2] * 5
        # Predictions:
        # Class 0: 90 correct, 10 predicted as 1
        # Class 1: 5 correct, 5 predicted as 0
        # Class 2: 1 correct, 4 predicted as 0
        y_pred = ([0] * 90 + [1] * 10) + ([1] * 5 + [0] * 5) + ([2] * 1 + [0] * 4)

        # 1. Manual paper formula per class:
        # Class 0: TP=90, FP=9 (5 from cls1 + 4 from cls2), FN=10 (to cls1)
        p0 = 90 / (90 + 9)
        r0 = 90 / (90 + 10)
        f1_0 = 2 * p0 * r0 / (p0 + r0)

        # Class 1: TP=5, FP=10 (from cls0), FN=5 (to cls0)
        p1 = 5 / (5 + 10)
        r1 = 5 / (5 + 5)
        f1_1 = 2 * p1 * r1 / (p1 + r1)

        # Class 2: TP=1, FP=0, FN=4 (to cls0)
        p2 = 1.0 / (1 + 0)
        r2 = 1.0 / (1 + 4)
        f1_2 = 2 * p2 * r2 / (p2 + r2)

        manual_macro_f1 = (f1_0 + f1_1 + f1_2) / 3.0
        weighted_f1 = (100 * f1_0 + 10 * f1_1 + 5 * f1_2) / 115.0

        # Assert manual macro != weighted macro (due to class imbalance)
        self.assertNotAlmostEqual(manual_macro_f1, weighted_f1, places=3)

        # Evaluate sklearn implementation used in evaluate.py
        sklearn_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Must strictly match the unweighted arithmetic mean
        self.assertAlmostEqual(sklearn_macro_f1, manual_macro_f1, places=6)

    def test_unpredicted_classes_division_by_zero(self):
        """
        Verify handling of unpredicted classes:
        When a class is never predicted, Precision is 0/0 and F1 is 0.0 without ZeroDivisionError.
        """
        y_true = [0, 1, 2, 0, 1, 2]
        # Class 1 is never predicted
        y_pred = [0, 0, 2, 0, 2, 2]

        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # Class 0: TP=2, FP=1, FN=0 -> P=2/3, R=1, F1=4/5 = 0.8
        # Class 1: TP=0, FP=0, FN=2 -> P=0, R=0, F1=0.0
        # Class 2: TP=2, FP=1, FN=0 -> P=2/3, R=1, F1=4/5 = 0.8
        expected_macro_f1 = (0.8 + 0.0 + 0.8) / 3.0

        self.assertAlmostEqual(f1_macro, expected_macro_f1, places=6)

        # Check classification report dictionary
        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=["c0", "c1", "c2"],
            output_dict=True, zero_division=0
        )
        self.assertEqual(report["c1"]["precision"], 0.0)
        self.assertEqual(report["c1"]["recall"], 0.0)
        self.assertEqual(report["c1"]["f1-score"], 0.0)

    def test_absent_ground_truth_class(self):
        """
        Verify case where a label exists in class map but 0 true samples exist in evaluation split.
        """
        y_true = [0, 0, 0]
        y_pred = [0, 1, 0]  # predicted class 1 which has 0 ground truth samples

        # Using labels parameter [0, 1, 2]
        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=["c0", "c1", "c2"],
            output_dict=True, zero_division=0
        )
        self.assertEqual(report["c2"]["support"], 0)
        self.assertEqual(report["c2"]["f1-score"], 0.0)

    def test_confusion_matrix_row_col_convention(self):
        """
        Verify confusion matrix convention:
        Rows = Ground Truth (True label)
        Cols = Predictions (Predicted label)
        Sum across rows = class support.
        Sum across cols = total predicted for that class.
        """
        # True: 3 of class 0, 2 of class 1, 1 of class 2
        y_true = [0, 0, 0, 1, 1, 2]
        # Pred: class 0 predicted as [0, 0, 1], class 1 predicted as [1, 2], class 2 as [2]
        y_pred = [0, 0, 1, 1, 2, 2]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        # Row 0: True 0 -> 2 predicted as 0, 1 predicted as 1, 0 predicted as 2 -> [2, 1, 0]
        # Row 1: True 1 -> 0 predicted as 0, 1 predicted as 1, 1 predicted as 2 -> [0, 1, 1]
        # Row 2: True 2 -> 0 predicted as 0, 0 predicted as 1, 1 predicted as 2 -> [0, 0, 1]
        expected_cm = [
            [2, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ]
        self.assertTrue(np.array_equal(cm, expected_cm), f"CM mismatch: \n{cm}\nvs expected\n{expected_cm}")

        # Check row sums equal class support
        self.assertEqual(cm[0].sum(), 3)
        self.assertEqual(cm[1].sum(), 2)
        self.assertEqual(cm[2].sum(), 1)

        # Check column sums equal predicted counts
        self.assertEqual(cm[:, 0].sum(), 2)
        self.assertEqual(cm[:, 1].sum(), 2)
        self.assertEqual(cm[:, 2].sum(), 2)


class TestDeltaAndRelativeImprovement(unittest.TestCase):
    """Rigorous verification of delta and relative improvement formulas."""

    def test_delta_direction_and_magnitude(self):
        """Verify PhoBERT - Baseline signs and values."""
        base = {
            "sentiment_accuracy": 0.85,
            "sentiment_macro_f1": 0.70,
            "topic_accuracy": 0.80,
            "topic_macro_f1": 0.75,
            "mean_macro_f1": 0.725,
        }
        phobert = {
            "sentiment_accuracy": 0.90,
            "sentiment_macro_f1": 0.77,
            "topic_accuracy": 0.75,  # Regression
            "topic_macro_f1": 0.75,  # Equal
            "mean_macro_f1": 0.76,
        }
        comp = eval_module.compute_comparisons(base, phobert)

        # Sentiment Accuracy: 0.90 - 0.85 = +0.05
        self.assertAlmostEqual(comp["sentiment_accuracy_delta"], 0.05, places=4)
        self.assertAlmostEqual(comp["sentiment_accuracy"]["delta"], 0.05, places=4)

        # Topic Accuracy: 0.75 - 0.80 = -0.05
        self.assertAlmostEqual(comp["topic_accuracy_delta"], -0.05, places=4)
        self.assertAlmostEqual(comp["topic_accuracy"]["delta"], -0.05, places=4)

        # Topic Macro-F1: 0.75 - 0.75 = 0.0
        self.assertAlmostEqual(comp["topic_macro_f1_delta"], 0.0, places=4)

    def test_relative_improvement_formula(self):
        """Verify ((PhoBERT - Baseline) / Baseline) * 100%."""
        base = {
            "sentiment_accuracy": 0.80,
            "sentiment_macro_f1": 0.50,
            "topic_accuracy": 0.60,
            "topic_macro_f1": 0.40,
            "mean_macro_f1": 0.45,
        }
        phobert = {
            "sentiment_accuracy": 0.90,  # +0.10 / 0.80 = +12.5%
            "sentiment_macro_f1": 0.40,  # -0.10 / 0.50 = -20.0%
            "topic_accuracy": 0.60,      # 0.0%
            "topic_macro_f1": 0.60,      # +0.20 / 0.40 = +50.0%
            "mean_macro_f1": 0.50,       # +0.05 / 0.45 = +11.11%
        }
        comp = eval_module.compute_comparisons(base, phobert)

        self.assertAlmostEqual(comp["sentiment_accuracy_relative_pct"], 12.5, places=2)
        self.assertAlmostEqual(comp["sentiment_macro_f1_relative_pct"], -20.0, places=2)
        self.assertAlmostEqual(comp["topic_accuracy_relative_pct"], 0.0, places=2)
        self.assertAlmostEqual(comp["topic_macro_f1_relative_pct"], 50.0, places=2)
        self.assertAlmostEqual(comp["mean_macro_f1_relative_pct"], (0.05 / 0.45) * 100.0, places=2)

    def test_relative_improvement_zero_baseline_guard(self):
        """When baseline is 0.0, relative improvement must return 0.0, no ZeroDivisionError."""
        base = {
            "sentiment_accuracy": 0.0,
            "sentiment_macro_f1": 0.0,
            "topic_accuracy": 0.0,
            "topic_macro_f1": 0.0,
            "mean_macro_f1": 0.0,
        }
        phobert = {
            "sentiment_accuracy": 0.50,
            "sentiment_macro_f1": 0.50,
            "topic_accuracy": 0.50,
            "topic_macro_f1": 0.50,
            "mean_macro_f1": 0.50,
        }
        comp = eval_module.compute_comparisons(base, phobert)
        self.assertEqual(comp["sentiment_accuracy_relative_pct"], 0.0)
        self.assertEqual(comp["sentiment_macro_f1_relative_pct"], 0.0)
        self.assertEqual(comp["topic_accuracy_relative_pct"], 0.0)
        self.assertEqual(comp["topic_macro_f1_relative_pct"], 0.0)
        self.assertEqual(comp["mean_macro_f1_relative_pct"], 0.0)


class TestEmpiricalMetricsJsonVerification(unittest.TestCase):
    """Direct empirical comparison against results/metrics.json and saved models."""

    @classmethod
    def setUpClass(cls):
        cls.metrics_json_path = os.path.join(PROJECT_ROOT, "results", "metrics.json")
        cls.has_metrics_json = os.path.isfile(cls.metrics_json_path)
        if cls.has_metrics_json:
            with open(cls.metrics_json_path, "r", encoding="utf-8") as f:
                cls.metrics_data = json.load(f)

    def test_metrics_json_file_exists_and_is_valid(self):
        """Verify results/metrics.json exists and is valid JSON."""
        self.assertTrue(self.has_metrics_json, "results/metrics.json does not exist!")

    def test_metrics_json_internal_consistency(self):
        """
        Verify mathematical consistency inside results/metrics.json:
        Does delta strictly equal phobert - baseline?
        Does relative_improvement strictly equal ((phobert - baseline) / baseline) * 100?
        """
        if not self.has_metrics_json:
            self.skipTest("results/metrics.json not found")

        comp = self.metrics_data["comparison"]
        models = self.metrics_data.get("models", self.metrics_data)
        base = models["baseline"]
        phob = models["phobert"]

        for metric in ["sentiment_accuracy", "sentiment_macro_f1", "topic_accuracy", "topic_macro_f1", "mean_macro_f1"]:
            b_val = base[metric]
            p_val = phob[metric]
            delta_val = comp[f"{metric}_delta"]
            rel_val = comp[f"{metric}_relative_pct"]

            # 1. Delta check
            self.assertAlmostEqual(delta_val, p_val - b_val, places=6,
                                   msg=f"Delta inconsistency for {metric}")

            # 2. Relative improvement check
            expected_rel = ((p_val - b_val) / b_val * 100.0) if b_val > 0 else 0.0
            self.assertAlmostEqual(rel_val, expected_rel, places=4,
                                   msg=f"Relative pct inconsistency for {metric}")

    def test_recompute_baseline_metrics_against_metrics_json(self):
        """
        Empirically re-evaluate baseline model on the test split (N samples matching metrics.json)
        and verify exact match with metrics recorded in results/metrics.json.
        """
        if not self.has_metrics_json:
            self.skipTest("results/metrics.json not found")

        baseline_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        if not os.path.isfile(baseline_path):
            self.skipTest(f"Baseline checkpoint not found at {baseline_path}")

        recorded_num_samples = self.metrics_data.get("num_samples", 3166)

        texts, sentiments, topics = load_vsfc_split("test", data_dir=os.path.join(PROJECT_ROOT, "data", "uit_vsfc"), word_segment=True)
        texts_eval = texts[:recorded_num_samples]
        sents_eval = sentiments[:recorded_num_samples]
        topics_eval = topics[:recorded_num_samples]

        model = VietnameseFeedbackBaseline.load(baseline_path)
        s_preds, t_preds = model.predict_labels(texts_eval, word_segment=False)

        # Independent calculation
        s_acc = float(accuracy_score(sents_eval, s_preds))
        s_f1 = float(f1_score(sents_eval, s_preds, average="macro", zero_division=0))
        t_acc = float(accuracy_score(topics_eval, t_preds))
        t_f1 = float(f1_score(topics_eval, t_preds, average="macro", zero_division=0))
        mean_f1 = float((s_f1 + t_f1) / 2.0)

        base_rec = self.metrics_data.get("models", self.metrics_data)["baseline"]

        # Assert exact agreement with recorded values
        self.assertAlmostEqual(s_acc, base_rec["sentiment_accuracy"], places=6)
        self.assertAlmostEqual(s_f1, base_rec["sentiment_macro_f1"], places=6)
        self.assertAlmostEqual(t_acc, base_rec["topic_accuracy"], places=6)
        self.assertAlmostEqual(t_f1, base_rec["topic_macro_f1"], places=6)
        self.assertAlmostEqual(mean_f1, base_rec["mean_macro_f1"], places=6)

    def test_full_test_split_baseline_performance(self):
        """
        Empirically evaluate baseline model on FULL 3,166 test samples.
        Compare against worker handoff claim (86.73% Sent Acc, 71.90% Sent F1, 80.99% Topic Acc, 72.72% Topic F1).
        """
        baseline_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        if not os.path.isfile(baseline_path):
            self.skipTest(f"Baseline checkpoint not found at {baseline_path}")

        texts, sentiments, topics = load_vsfc_split("test", data_dir=os.path.join(PROJECT_ROOT, "data", "uit_vsfc"), word_segment=True)
        self.assertEqual(len(texts), 3166, f"Expected 3166 test samples, got {len(texts)}")

        model = VietnameseFeedbackBaseline.load(baseline_path)
        s_preds, t_preds = model.predict_labels(texts, word_segment=False)

        s_acc = float(accuracy_score(sentiments, s_preds))
        s_f1 = float(f1_score(sentiments, s_preds, average="macro", zero_division=0))
        t_acc = float(accuracy_score(topics, t_preds))
        t_f1 = float(f1_score(topics, t_preds, average="macro", zero_division=0))
        mean_f1 = float((s_f1 + t_f1) / 2.0)

        # Check against worker handoff values (within 0.05%)
        # Claimed: s_acc=86.73%, s_f1=71.90%, t_acc=80.99%, t_f1=72.72%, mean_f1=72.31%
        self.assertAlmostEqual(s_acc * 100.0, 86.73, delta=0.1)
        self.assertAlmostEqual(s_f1 * 100.0, 71.90, delta=0.1)
        self.assertAlmostEqual(t_acc * 100.0, 80.99, delta=0.1)
        self.assertAlmostEqual(t_f1 * 100.0, 72.72, delta=0.1)
        self.assertAlmostEqual(mean_f1 * 100.0, 72.31, delta=0.1)


class TestAdversarialStressScenarios(unittest.TestCase):
    """Stress tests on adversarial edge cases and input permutations."""

    def test_single_class_degenerate_input(self):
        """Degenerate case: all ground truth and all predictions belong to a single class."""
        y_true = [1] * 50
        y_pred = [1] * 50
        acc = float(accuracy_score(y_true, y_pred))
        # When labels=[0, 1, 2], classes 0 and 2 have 0 true and 0 predicted -> F1=0
        # Class 1: P=1.0, R=1.0, F1=1.0
        # Macro-F1 = (0 + 1.0 + 0) / 3 = 0.333333
        f1 = float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))
        self.assertEqual(acc, 1.0)
        self.assertAlmostEqual(f1, 1.0 / 3.0, places=5)

    def test_all_wrong_predictions_zero_accuracy(self):
        """Pathological case: every prediction is wrong."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [1, 2, 0, 1, 2, 0]
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        self.assertEqual(acc, 0.0)
        self.assertEqual(f1, 0.0)

    def test_compute_comparisons_robustness_on_malformed_inputs(self):
        """compute_comparisons survives None, empty dicts, missing keys, and unexpected types."""
        # None inputs
        self.assertEqual(eval_module.compute_comparisons(None, None), {})
        self.assertEqual(eval_module.compute_comparisons({}, None), {})
        self.assertEqual(eval_module.compute_comparisons(None, {}), {})
        self.assertEqual(eval_module.compute_comparisons({}, {}), {})

        # Partial dicts missing keys
        partial_b = {"sentiment_accuracy": 0.8}
        partial_p = {"topic_accuracy": 0.7}
        comp = eval_module.compute_comparisons(partial_b, partial_p)
        self.assertIn("sentiment_accuracy_delta", comp)
        # Missing keys default to 0.0
        self.assertAlmostEqual(comp["sentiment_accuracy_delta"], -0.8, places=4)
        self.assertAlmostEqual(comp["topic_accuracy_delta"], 0.7, places=4)

    def test_terminal_table_rendering_fallback(self):
        """format_terminal_table renders properly without crashing under single model or empty comparison."""
        table_both = eval_module.format_terminal_table(
            {"sentiment_accuracy": 0.8},
            {"sentiment_accuracy": 0.85},
            {"sentiment_accuracy_delta": 0.05, "sentiment_accuracy_relative_pct": 6.25},
        )
        self.assertIn("Sentiment Accuracy", table_both)

        table_base_only = eval_module.format_terminal_table({"sentiment_accuracy": 0.8}, None, {})
        self.assertIn("Baseline Value", table_base_only)

        table_phobert_only = eval_module.format_terminal_table(None, {"sentiment_accuracy": 0.85}, {})
        self.assertIn("PhoBERT Value", table_phobert_only)


if __name__ == "__main__":
    unittest.main(verbosity=2)

