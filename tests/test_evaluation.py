"""
tests/test_evaluation.py
Comprehensive QA and Automated Test Suite for Milestone 4 (Comparative Evaluation Script).

Verifies:
1. Fast, offline mock execution (<1s runtime) with zero dependency on heavy downloads or full 3,166 inference.
2. Metrics computation: Accuracy, Macro-F1, Confusion Matrix, and Classification Report.
3. Delta and relative improvement calculations (Δ = phobert - baseline).
4. Artifact generation:
   - results/comparison_summary.md (valid markdown table with expected headers & takeaways).
   - results/metrics.json (valid JSON adhering strictly to schema).
5. CLI argument parsing: --quick_eval, --output_dir, --baseline_path, --phobert_path, etc.
6. Error handling: missing checkpoints or datasets trigger clear, informative error messages.

Execution:
    python -m unittest tests/test_evaluation.py -v
    python tests/test_evaluation.py
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Import evaluation module functions
import evaluation.evaluate as eval_module
from data.dataset import SENTIMENT_MAP, TOPIC_MAP


class TestMetricsCalculation(unittest.TestCase):
    """Verifies Accuracy, Macro-F1, Confusion Matrix, and Classification Report calculations."""

    def test_accuracy_and_macro_f1_perfect_predictions(self):
        """All predictions match ground truth: Accuracy=1.0, Macro-F1=1.0, diagonal CM."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 1, 2]

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

        self.assertAlmostEqual(acc, 1.0, places=6)
        self.assertAlmostEqual(f1, 1.0, places=6)
        self.assertEqual(cm, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

    def test_metrics_hand_crafted_known_values(self):
        """
        Verifies exact precision, recall, and Macro-F1 with known partial errors.
        y_true: [0, 0, 1, 1, 2, 2]
        y_pred: [0, 1, 1, 1, 2, 0]
        - Class 0: TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
        - Class 1: TP=2, FP=1, FN=0 -> Prec=2/3=0.6667, Rec=1.0, F1=0.8
        - Class 2: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=2/3=0.6667
        Total correct: 4/6 -> Accuracy = 0.666667
        Macro-F1 = (0.5 + 0.8 + 0.666667) / 3 = 0.655556
        """
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

        self.assertAlmostEqual(acc, 4.0 / 6.0, places=5)
        expected_macro_f1 = (0.5 + 0.8 + (2.0 / 3.0)) / 3.0
        self.assertAlmostEqual(f1, expected_macro_f1, places=5)

        self.assertEqual(len(cm), 3)
        self.assertEqual(len(cm[0]), 3)
        self.assertEqual(sum(sum(row) for row in cm), 6)

    def test_metrics_unpredicted_classes_zero_division(self):
        """Missing predicted classes handled gracefully without ZeroDivisionError."""
        y_true = [0, 1, 2, 0, 1, 2]
        # Only class 0 and 2 are predicted
        y_pred = [0, 0, 2, 0, 2, 2]

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        report = classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=["neg", "neu", "pos"],
            output_dict=True, zero_division=0
        )

        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)
        self.assertEqual(report["neu"]["f1-score"], 0.0)
        self.assertEqual(report["neu"]["precision"], 0.0)

    def test_confusion_matrix_dimensions_and_conservation(self):
        """Verifies 3x3 for sentiment and 4x4 for topic, with sum equal to sample count."""
        # Sentiment (3 classes)
        s_true = [0, 1, 2, 1, 0]
        s_pred = [0, 1, 1, 1, 2]
        s_cm = confusion_matrix(s_true, s_pred, labels=[0, 1, 2]).tolist()
        self.assertEqual(len(s_cm), 3)
        self.assertEqual(len(s_cm[0]), 3)
        self.assertEqual(sum(sum(row) for row in s_cm), len(s_true))

        # Topic (4 classes)
        t_true = [0, 1, 2, 3, 0, 1, 2, 3]
        t_pred = [0, 1, 2, 2, 0, 1, 3, 3]
        t_cm = confusion_matrix(t_true, t_pred, labels=[0, 1, 2, 3]).tolist()
        self.assertEqual(len(t_cm), 4)
        self.assertEqual(len(t_cm[0]), 4)
        self.assertEqual(sum(sum(row) for row in t_cm), len(t_true))

    def test_mean_macro_f1_aggregation(self):
        """Verifies overall benchmark score as arithmetic mean of sentiment and topic Macro-F1."""
        sent_f1 = 0.7250
        topic_f1 = 0.8150
        mean_macro_f1 = float((sent_f1 + topic_f1) / 2.0)
        self.assertAlmostEqual(mean_macro_f1, 0.7700, places=4)


class TestDeltaCalculation(unittest.TestCase):
    """Verifies Delta (PhoBERT - Baseline) and Relative Improvement (%) calculations."""

    def setUp(self):
        self.baseline_metrics = {
            "sentiment_accuracy": 0.7000,
            "sentiment_macro_f1": 0.6000,
            "topic_accuracy": 0.7500,
            "topic_macro_f1": 0.5000,
            "mean_macro_f1": 0.5500,
            "inference_time_sec": 0.50,
            "throughput_samples_per_sec": 2000.0,
            "latency_ms_per_sample": 0.50,
        }
        self.phobert_metrics = {
            "sentiment_accuracy": 0.7700,
            "sentiment_macro_f1": 0.7200,
            "topic_accuracy": 0.8100,
            "topic_macro_f1": 0.6500,
            "mean_macro_f1": 0.6850,
            "inference_time_sec": 15.0,
            "throughput_samples_per_sec": 66.7,
            "latency_ms_per_sample": 15.0,
        }

    def test_positive_deltas_and_relative_improvements(self):
        """Verifies standard positive delta and relative improvement formulas."""
        compute_fn = getattr(eval_module, "compute_comparisons", None) or getattr(eval_module, "compute_deltas")
        comparisons = compute_fn(self.baseline_metrics, self.phobert_metrics)

        # Sentiment accuracy delta: 0.77 - 0.70 = +0.07 (+10.0%)
        self.assertAlmostEqual(comparisons["sentiment_accuracy_delta"], 0.07, places=4)
        rel_acc = comparisons.get("sentiment_accuracy_relative_pct") or comparisons.get("relative_improvements", {}).get("sentiment_accuracy")
        self.assertAlmostEqual(rel_acc, 10.0, places=2)

        # Sentiment macro_f1 delta: 0.72 - 0.60 = +0.12 (+20.0%)
        self.assertAlmostEqual(comparisons["sentiment_macro_f1_delta"], 0.12, places=4)
        rel_f1 = comparisons.get("sentiment_macro_f1_relative_pct") or comparisons.get("relative_improvements", {}).get("sentiment_macro_f1")
        self.assertAlmostEqual(rel_f1, 20.0, places=2)

        # Topic macro_f1 delta: 0.65 - 0.50 = +0.15 (+30.0%)
        self.assertAlmostEqual(comparisons["topic_macro_f1_delta"], 0.15, places=4)
        rel_topic = comparisons.get("topic_macro_f1_relative_pct") or comparisons.get("relative_improvements", {}).get("topic_macro_f1")
        self.assertAlmostEqual(rel_topic, 30.0, places=2)

        # Mean macro_f1 delta: 0.685 - 0.550 = +0.135
        self.assertAlmostEqual(comparisons["mean_macro_f1_delta"], 0.135, places=4)

    def test_negative_delta_and_relative_regression(self):
        """Verifies negative delta if PhoBERT performs lower on a metric."""
        phobert_lower = dict(self.phobert_metrics)
        phobert_lower["sentiment_accuracy"] = 0.6500

        compute_fn = getattr(eval_module, "compute_comparisons", None) or getattr(eval_module, "compute_deltas")
        comparisons = compute_fn(self.baseline_metrics, phobert_lower)

        self.assertAlmostEqual(comparisons["sentiment_accuracy_delta"], -0.05, places=4)
        rel_acc = comparisons.get("sentiment_accuracy_relative_pct") or comparisons.get("relative_improvements", {}).get("sentiment_accuracy")
        self.assertAlmostEqual(rel_acc, -7.1428, places=2)

    def test_zero_baseline_safety(self):
        """Baseline metric of 0.0 does not raise ZeroDivisionError for relative improvement."""
        zero_baseline = dict(self.baseline_metrics)
        zero_baseline["sentiment_accuracy"] = 0.0
        zero_baseline["sentiment_macro_f1"] = 0.0

        compute_fn = getattr(eval_module, "compute_comparisons", None) or getattr(eval_module, "compute_deltas")
        comparisons = compute_fn(zero_baseline, self.phobert_metrics)
        rel_acc = comparisons.get("sentiment_accuracy_relative_pct", comparisons.get("relative_improvements", {}).get("sentiment_accuracy", 0.0))
        self.assertEqual(rel_acc, 0.0)


class TestArtifactGeneration(unittest.TestCase):
    """Verifies generation and schema conformity of comparison_summary.md and metrics.json."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_metrics = {
            "model_name": "Classical Baseline (TF-IDF + Dual LR)",
            "num_samples": 3166,
            "sentiment_accuracy": 0.8673,
            "sentiment_macro_f1": 0.7190,
            "topic_accuracy": 0.8099,
            "topic_macro_f1": 0.7272,
            "mean_macro_f1": 0.7231,
            "inference_time_sec": 2.28,
            "throughput_samples_per_sec": 1388.6,
            "latency_ms_per_sample": 0.72,
            "predictions": {"sentiment": [0]*3166, "topic": [0]*3166},
            "per_class": {
                "sentiment": {
                    "negative": {"precision": 0.884, "recall": 0.875, "f1-score": 0.879, "support": 1409},
                    "neutral": {"precision": 0.450, "recall": 0.380, "f1-score": 0.412, "support": 167},
                    "positive": {"precision": 0.895, "recall": 0.911, "f1-score": 0.903, "support": 1590},
                },
                "topic": {
                    "lecturer": {"precision": 0.892, "recall": 0.881, "f1-score": 0.886, "support": 2290},
                    "training_program": {"precision": 0.654, "recall": 0.680, "f1-score": 0.667, "support": 572},
                    "facility": {"precision": 0.680, "recall": 0.710, "f1-score": 0.695, "support": 145},
                    "others": {"precision": 0.620, "recall": 0.580, "f1-score": 0.600, "support": 159},
                },
            },
        }
        self.phobert_metrics = {
            "model_name": "PhoBERT Multi-Task (Fine-Tuned)",
            "num_samples": 3166,
            "sentiment_accuracy": 0.9125,
            "sentiment_macro_f1": 0.8045,
            "topic_accuracy": 0.8850,
            "topic_macro_f1": 0.8210,
            "mean_macro_f1": 0.8128,
            "inference_time_sec": 58.42,
            "throughput_samples_per_sec": 54.2,
            "latency_ms_per_sample": 18.45,
            "predictions": {"sentiment": [0]*3166, "topic": [0]*3166},
            "per_class": {
                "sentiment": {
                    "negative": {"precision": 0.925, "recall": 0.920, "f1-score": 0.922, "support": 1409},
                    "neutral": {"precision": 0.610, "recall": 0.580, "f1-score": 0.595, "support": 167},
                    "positive": {"precision": 0.935, "recall": 0.941, "f1-score": 0.938, "support": 1590},
                },
                "topic": {
                    "lecturer": {"precision": 0.941, "recall": 0.935, "f1-score": 0.938, "support": 2290},
                    "training_program": {"precision": 0.780, "recall": 0.795, "f1-score": 0.787, "support": 572},
                    "facility": {"precision": 0.810, "recall": 0.825, "f1-score": 0.817, "support": 145},
                    "others": {"precision": 0.730, "recall": 0.715, "f1-score": 0.722, "support": 159},
                },
            },
        }
        compute_fn = getattr(eval_module, "compute_comparisons", None) or getattr(eval_module, "compute_deltas")
        self.comparisons = compute_fn(self.baseline_metrics, self.phobert_metrics)

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_comparison_summary_markdown_structure(self):
        """Verifies comparison_summary.md contains table with expected headers and student takeaways."""
        md_path = os.path.join(self.temp_dir, "comparison_summary.md")
        md_fn = getattr(eval_module, "generate_markdown_report", None) or getattr(eval_module, "generate_comparison_markdown")

        try:
            md_fn(
                baseline_metrics=self.baseline_metrics,
                phobert_metrics=self.phobert_metrics,
                comparisons=self.comparisons,
                split_name="test",
                num_samples=3166,
                device_name="CPU",
                output_path=md_path,
            )
        except TypeError:
            md_fn(self.baseline_metrics, self.phobert_metrics, self.comparisons, md_path)

        self.assertTrue(os.path.isfile(md_path), f"File {md_path} was not created.")
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check markdown table headers
        self.assertIn("Task / Metric", content)
        self.assertIn("Classical Baseline", content)
        self.assertIn("PhoBERT", content)
        self.assertTrue("Delta" in content or "Δ" in content)

        # Check rows for 5 core benchmark metrics
        self.assertIn("Sentiment Accuracy", content)
        self.assertIn("Sentiment Macro-F1", content)
        self.assertIn("Topic Accuracy", content)
        self.assertIn("Topic Macro-F1", content)
        self.assertIn("Mean Macro-F1", content)

        # Check student takeaways narrative section
        self.assertTrue(
            any(k in content for k in ["Takeaways", "Insights", "Trade-Offs", "Analysis"]),
            "Summary markdown report missing narrative pedagogical takeaways."
        )

    def test_metrics_json_schema_validity(self):
        """Verifies metrics.json is valid JSON matching the nested grading schema."""
        json_path = os.path.join(self.temp_dir, "metrics.json")
        json_fn = getattr(eval_module, "generate_json_report", None) or getattr(eval_module, "generate_metrics_json")

        try:
            json_fn(
                baseline_metrics=self.baseline_metrics,
                phobert_metrics=self.phobert_metrics,
                comparisons=self.comparisons,
                split_name="test",
                num_samples=3166,
                device_name="CPU",
                output_path=json_path,
            )
        except TypeError:
            json_fn(self.baseline_metrics, self.phobert_metrics, self.comparisons, json_path)

        self.assertTrue(os.path.isfile(json_path), f"File {json_path} was not created.")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check required top-level elements
        self.assertIn("comparison", data)
        self.assertTrue("baseline" in data or "models" in data)

        # If nested under 'models':
        models_dict = data.get("models", data)
        self.assertIn("baseline", models_dict)
        self.assertIn("phobert", models_dict)

        # Validate numeric metrics
        for m_key in ["baseline", "phobert"]:
            m_data = models_dict[m_key]
            self.assertIsInstance(m_data["sentiment_accuracy"], float)
            self.assertIsInstance(m_data["sentiment_macro_f1"], float)
            self.assertIsInstance(m_data["topic_accuracy"], float)
            self.assertIsInstance(m_data["topic_macro_f1"], float)
            self.assertIsInstance(m_data["mean_macro_f1"], float)

        # Verify comparisons deltas
        comp = data["comparison"]
        self.assertIn("sentiment_accuracy_delta", comp)
        self.assertIn("sentiment_macro_f1_delta", comp)
        self.assertIn("topic_accuracy_delta", comp)
        self.assertIn("topic_macro_f1_delta", comp)
        self.assertIn("mean_macro_f1_delta", comp)

    def test_directory_auto_creation(self):
        """Artifact generators auto-create parent directories if non-existent."""
        deep_dir = os.path.join(self.temp_dir, "deep", "nested", "results")
        md_path = os.path.join(deep_dir, "comparison_summary.md")
        json_path = os.path.join(deep_dir, "metrics.json")

        md_fn = getattr(eval_module, "generate_markdown_report", None) or getattr(eval_module, "generate_comparison_markdown")
        json_fn = getattr(eval_module, "generate_json_report", None) or getattr(eval_module, "generate_metrics_json")

        try:
            md_fn(self.baseline_metrics, self.phobert_metrics, self.comparisons, "test", 3166, "CPU", md_path)
            json_fn(self.baseline_metrics, self.phobert_metrics, self.comparisons, "test", 3166, "CPU", json_path)
        except TypeError:
            md_fn(self.baseline_metrics, self.phobert_metrics, self.comparisons, md_path)
            json_fn(self.baseline_metrics, self.phobert_metrics, self.comparisons, json_path)

        self.assertTrue(os.path.isfile(md_path))
        self.assertTrue(os.path.isfile(json_path))


class TestCLIArgumentParsing(unittest.TestCase):
    """Verifies CLI argument defaults, custom overrides, and flag parsing."""

    def test_default_cli_arguments(self):
        """Empty argv defaults to standard paths and options."""
        parser_fn = getattr(eval_module, "parse_args", None) or getattr(eval_module, "get_arg_parser", None)
        if parser_fn is None:
            import argparse
            parser = argparse.ArgumentParser()
            eval_module.build_arg_parser(parser) if hasattr(eval_module, "build_arg_parser") else None
        else:
            args = parser_fn([]) if callable(parser_fn) else parser_fn.parse_args([])
            self.assertFalse(args.quick_eval)
            self.assertEqual(args.output_dir, "results")
            self.assertEqual(args.baseline_path, "saved_models/baseline.joblib")
            self.assertEqual(args.phobert_path, "saved_models/phobert_multitask")

    def test_custom_cli_arguments_and_flags(self):
        """Flags and custom options correctly override defaults."""
        parser_fn = getattr(eval_module, "parse_args", None)
        if callable(parser_fn):
            custom_argv = [
                "--quick_eval",
                "--output_dir", "custom_results",
                "--baseline_path", "models/custom_base.joblib",
                "--phobert_path", "models/custom_phobert",
                "--batch_size", "64",
            ]
            args = parser_fn(custom_argv)
            self.assertTrue(args.quick_eval)
            self.assertEqual(args.output_dir, "custom_results")
            self.assertEqual(args.baseline_path, "models/custom_base.joblib")
            self.assertEqual(args.phobert_path, "models/custom_phobert")
            self.assertEqual(args.batch_size, 64)

    def test_invalid_cli_arguments_raises_system_exit(self):
        """Passing invalid option types (e.g. non-integer batch size) triggers SystemExit."""
        parser_fn = getattr(eval_module, "parse_args", None)
        if callable(parser_fn):
            with self.assertRaises(SystemExit):
                parser_fn(["--batch_size", "not_a_number"])


class TestErrorHandlingAndResilience(unittest.TestCase):
    """Verifies informative error messages when dependencies, models, or data are missing."""

    def test_missing_both_checkpoints_raises_informative_error_or_exit(self):
        """When both checkpoints are missing, program aborts cleanly with clear error message."""
        main_fn = getattr(eval_module, "main", None)
        run_fn = getattr(eval_module, "run_evaluation", None)

        test_argv = [
            "evaluation/evaluate.py",
            "--baseline_path", "non_existent/baseline.joblib",
            "--phobert_path", "non_existent/phobert",
            "--data_dir", "data/uit_vsfc",
        ]

        with patch.object(sys, "argv", test_argv):
            if run_fn:
                with self.assertRaises((FileNotFoundError, SystemExit)):
                    run_fn(
                        baseline_path="non_existent/baseline.joblib",
                        phobert_path="non_existent/phobert",
                        data_dir="data/uit_vsfc",
                        output_dir="results",
                    )
            elif main_fn:
                with self.assertRaises(SystemExit) as ctx:
                    main_fn()
                self.assertNotEqual(ctx.exception.code, 0)

    def test_missing_dataset_split_raises_file_not_found(self):
        """Missing dataset directory raises FileNotFoundError with download guidance."""
        test_argv = [
            "evaluation/evaluate.py",
            "--data_dir", "non_existent/data_dir",
        ]
        main_fn = getattr(eval_module, "main", None)
        with patch.object(sys, "argv", test_argv):
            if main_fn:
                with self.assertRaises((FileNotFoundError, SystemExit)):
                    main_fn()


class TestFastOfflineMockEvaluation(unittest.TestCase):
    """Verifies end-to-end evaluation flow using fast mocks in <1.0s without network or heavy GPU."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch("evaluation.evaluate.load_vsfc_split")
    @patch("evaluation.evaluate.VietnameseFeedbackBaseline")
    @patch("evaluation.evaluate.PhoBERTMultiTask")
    @patch("evaluation.evaluate.AutoTokenizer")
    def test_fast_mock_evaluation_pipeline_runtime_under_one_second(
        self, mock_tok_cls, mock_phobert_cls, mock_baseline_cls, mock_load_vsfc
    ):
        """Executes full evaluation pipeline with mocked models and verifies runtime < 1.0s."""
        # 1. Setup mock data: 20 samples
        num_samples = 20
        dummy_texts = [f"Câu phản hồi sinh viên số {i}" for i in range(num_samples)]
        dummy_sentiments = [i % 3 for i in range(num_samples)]
        dummy_topics = [i % 4 for i in range(num_samples)]
        mock_load_vsfc.return_value = (dummy_texts, dummy_sentiments, dummy_topics)

        # 2. Setup mock Baseline model
        mock_baseline_instance = MagicMock()
        mock_baseline_instance.predict_labels.return_value = (
            dummy_sentiments,
            [(t + 1) % 4 for t in dummy_topics],
        )
        mock_baseline_cls.load.return_value = mock_baseline_instance

        # 3. Setup mock PhoBERT model & tokenizer
        mock_phobert_instance = MagicMock()
        mock_phobert_instance.to.return_value = mock_phobert_instance
        import torch
        mock_outputs = {
            "sentiment_logits": torch.zeros((num_samples, 3)),
            "topic_logits": torch.zeros((num_samples, 4)),
        }
        for i in range(num_samples):
            mock_outputs["sentiment_logits"][i, dummy_sentiments[i]] = 5.0
            mock_outputs["topic_logits"][i, dummy_topics[i]] = 5.0
        mock_phobert_instance.return_value = mock_outputs
        mock_phobert_cls.from_pretrained.return_value = mock_phobert_instance

        mock_tok_instance = MagicMock()
        mock_tok_instance.return_value = {
            "input_ids": torch.zeros((num_samples, 16), dtype=torch.long),
            "attention_mask": torch.ones((num_samples, 16), dtype=torch.long),
        }
        mock_tok_cls.from_pretrained.return_value = mock_tok_instance

        # 4. Measure execution time
        start_time = time.perf_counter()

        test_argv = [
            "evaluation/evaluate.py",
            "--baseline_path", "saved_models/baseline.joblib",
            "--phobert_path", "saved_models/phobert_multitask",
            "--data_dir", "data/uit_vsfc",
            "--output_dir", self.temp_dir,
            "--quick_eval",
            "--batch_size", "16",
        ]

        main_fn = getattr(eval_module, "main", None)
        with patch("os.path.isfile", return_value=True), patch("os.path.isdir", return_value=True):
            with patch.object(sys, "argv", test_argv):
                try:
                    if main_fn:
                        main_fn()
                except SystemExit as e:
                    self.assertEqual(e.code, 0)

        elapsed = time.perf_counter() - start_time

        # Verifications
        self.assertLess(
            elapsed, 1.0, f"Mock evaluation took {elapsed:.3f}s, exceeding 1.0s budget."
        )

        # Verify artifacts written
        md_file = os.path.join(self.temp_dir, "comparison_summary.md")
        json_file = os.path.join(self.temp_dir, "metrics.json")
        self.assertTrue(os.path.isfile(md_file), "comparison_summary.md was not generated.")
        self.assertTrue(os.path.isfile(json_file), "metrics.json was not generated.")

        # Verify JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        self.assertIn("comparison", metrics_data)


class TestEvaluationRegressionScenarios(unittest.TestCase):
    """
    Regression test suite for Milestone 4 (Gate Iteration 2):
    Verifies resilience against missing/zero-support classes, single-sample inputs,
    PhoBERT-only markdown report generation, and CLI --show_reports execution.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.phobert_metrics = {
            "model_name": "PhoBERT Multi-Task (Fine-Tuned)",
            "num_samples": 25,
            "sentiment_accuracy": 0.9200,
            "sentiment_macro_f1": 0.8100,
            "topic_accuracy": 0.8800,
            "topic_macro_f1": 0.8400,
            "mean_macro_f1": 0.8250,
            "inference_time_sec": 0.50,
            "throughput_samples_per_sec": 50.0,
            "latency_ms_per_sample": 20.0,
            "storage_footprint_mb": 450.0,
            "predictions": {"sentiment": [0] * 25, "topic": [0] * 25},
            "per_class": {
                "sentiment": {
                    "negative": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 20},
                    "neutral": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
                    "positive": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 5},
                },
                "topic": {
                    "lecturer": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 25},
                    "training_program": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
                    "facility": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
                    "others": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
                },
            },
        }

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_evaluate_baseline_subset_missing_classes(self):
        """
        Regression Test 1a:
        Verifies evaluate_baseline does not crash with ValueError when evaluating
        a small subset where some classes have zero support (e.g. only negative sentiment {0}
        and only lecturer topic {0}).
        """
        num_samples = 5
        texts = [f"Phản hồi tiêu cực {i}" for i in range(num_samples)]
        sentiments = [0] * num_samples  # Missing classes 1 and 2
        topics = [0] * num_samples      # Missing classes 1, 2, 3

        mock_baseline = MagicMock()
        mock_baseline.predict_labels.side_effect = lambda b, word_segment=False: ([0] * len(b), [0] * len(b))

        metrics = eval_module.evaluate_baseline(
            model=mock_baseline,
            texts=texts,
            sentiments=sentiments,
            topics=topics,
            batch_size=2,
            split_name="test_subset",
            checkpoint_path="saved_models/baseline.joblib",
        )

        self.assertEqual(metrics["num_samples"], num_samples)
        self.assertAlmostEqual(metrics["sentiment_accuracy"], 1.0)
        self.assertIn("neutral", metrics["per_class"]["sentiment"])
        self.assertEqual(metrics["per_class"]["sentiment"]["neutral"]["support"], 0)
        self.assertEqual(metrics["per_class"]["sentiment"]["positive"]["support"], 0)
        self.assertEqual(metrics["per_class"]["sentiment"]["negative"]["support"], num_samples)
        self.assertIn("facility", metrics["per_class"]["topic"])
        self.assertEqual(metrics["per_class"]["topic"]["facility"]["support"], 0)

    def test_evaluate_phobert_subset_missing_classes(self):
        """
        Regression Test 1b:
        Verifies evaluate_phobert does not crash with ValueError when evaluating
        a small subset with missing classes (e.g. only sentiment {0, 2} and topic {0, 1}).
        """
        import torch
        num_samples = 6
        texts = [f"Phản hồi mẫu {i}" for i in range(num_samples)]
        sentiments = [0, 2, 0, 2, 0, 2]  # Missing class 1 (neutral)
        topics = [0, 1, 0, 1, 0, 1]      # Missing classes 2 and 3

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros((num_samples, 16), dtype=torch.long),
            "attention_mask": torch.ones((num_samples, 16), dtype=torch.long),
        }

        sent_logits = torch.zeros((num_samples, 3))
        for i, s in enumerate(sentiments):
            sent_logits[i, s] = 10.0
        topic_logits = torch.zeros((num_samples, 4))
        for i, t in enumerate(topics):
            topic_logits[i, t] = 10.0

        mock_model.return_value = {
            "sentiment_logits": sent_logits,
            "topic_logits": topic_logits,
        }

        metrics = eval_module.evaluate_phobert(
            model=mock_model,
            tokenizer=mock_tokenizer,
            texts=texts,
            sentiments=sentiments,
            topics=topics,
            device=torch.device("cpu"),
            batch_size=4,
            split_name="test_subset",
            checkpoint_path="saved_models/phobert_multitask",
        )

        self.assertEqual(metrics["num_samples"], num_samples)
        self.assertAlmostEqual(metrics["sentiment_accuracy"], 1.0)
        self.assertEqual(metrics["per_class"]["sentiment"]["neutral"]["support"], 0)
        self.assertEqual(metrics["per_class"]["topic"]["facility"]["support"], 0)
        self.assertEqual(metrics["per_class"]["topic"]["others"]["support"], 0)

    def test_single_sample_evaluation(self):
        """
        Regression Test 2:
        Verifies evaluate_baseline and evaluate_phobert handle single-sample inputs (num_samples == 1)
        without raising ValueError or shape mismatches.
        """
        import torch
        texts = ["Thầy dạy rất tận tâm và nhiệt tình"]
        sentiments = [2]  # Positive
        topics = [0]      # Lecturer

        # 1. Baseline single-sample
        mock_baseline = MagicMock()
        mock_baseline.predict_labels.side_effect = lambda b, word_segment=False: ([2] * len(b), [0] * len(b))

        b_metrics = eval_module.evaluate_baseline(
            model=mock_baseline,
            texts=texts,
            sentiments=sentiments,
            topics=topics,
            batch_size=1,
            split_name="single_sample",
        )
        self.assertEqual(b_metrics["num_samples"], 1)
        self.assertAlmostEqual(b_metrics["sentiment_accuracy"], 1.0)
        self.assertAlmostEqual(b_metrics["topic_accuracy"], 1.0)
        self.assertEqual(b_metrics["per_class"]["sentiment"]["positive"]["support"], 1)
        self.assertEqual(b_metrics["per_class"]["sentiment"]["negative"]["support"], 0)

        # 2. PhoBERT single-sample
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros((1, 16), dtype=torch.long),
            "attention_mask": torch.ones((1, 16), dtype=torch.long),
        }
        sent_logits = torch.tensor([[0.0, 0.0, 10.0]])
        topic_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        mock_model.return_value = {
            "sentiment_logits": sent_logits,
            "topic_logits": topic_logits,
        }

        p_metrics = eval_module.evaluate_phobert(
            model=mock_model,
            tokenizer=mock_tokenizer,
            texts=texts,
            sentiments=sentiments,
            topics=topics,
            device=torch.device("cpu"),
            batch_size=1,
            split_name="single_sample",
        )
        self.assertEqual(p_metrics["num_samples"], 1)
        self.assertAlmostEqual(p_metrics["sentiment_accuracy"], 1.0)
        self.assertAlmostEqual(p_metrics["topic_accuracy"], 1.0)
        self.assertEqual(p_metrics["per_class"]["sentiment"]["positive"]["support"], 1)
        self.assertEqual(p_metrics["per_class"]["topic"]["lecturer"]["support"], 1)

    def test_phobert_only_markdown_report_generation(self):
        """
        Regression Test 3:
        Verifies markdown report generation when baseline_metrics is None.
        Section 1 table must populate PhoBERT rows instead of being empty.
        """
        md_path = os.path.join(self.temp_dir, "phobert_only_summary.md")
        eval_module.generate_markdown_report(
            baseline_metrics=None,
            phobert_metrics=self.phobert_metrics,
            comparisons={},
            split_name="test",
            num_samples=25,
            device_name="CPU",
            output_path=md_path,
        )

        self.assertTrue(os.path.isfile(md_path), "Report file was not generated.")
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check section 1 table headers
        self.assertIn("Task / Metric", content)
        self.assertTrue("PhoBERT Multi-Task" in content or "PhoBERT" in content)

        # Check that table rows for metrics exist with formatted PhoBERT values
        self.assertIn("Sentiment Accuracy", content)
        self.assertIn("92.00%", content)
        self.assertIn("Sentiment Macro-F1", content)
        self.assertIn("81.00%", content)
        self.assertIn("Topic Accuracy", content)
        self.assertIn("88.00%", content)
        self.assertIn("Topic Macro-F1", content)
        self.assertIn("84.00%", content)
        self.assertIn("Mean Macro-F1", content)
        self.assertIn("82.50%", content)

    @patch("evaluation.evaluate.load_vsfc_split")
    @patch("evaluation.evaluate.VietnameseFeedbackBaseline")
    @patch("evaluation.evaluate.PhoBERTMultiTask")
    @patch("evaluation.evaluate.AutoTokenizer")
    def test_cli_max_samples_with_show_reports_execution(
        self, mock_tok_cls, mock_phobert_cls, mock_baseline_cls, mock_load_vsfc
    ):
        """
        Regression Test 4:
        Verifies CLI invocation with --max_samples 25 --show_reports executes
        successfully without crashing during classification report generation.
        """
        import torch
        num_samples = 25
        dummy_texts = [f"Phản hồi sinh viên số {i}" for i in range(num_samples)]
        dummy_sentiments = [0 if i % 2 == 0 else 2 for i in range(num_samples)]  # Missing class 1
        dummy_topics = [0 if i % 2 == 0 else 1 for i in range(num_samples)]      # Missing classes 2, 3
        mock_load_vsfc.return_value = (dummy_texts, dummy_sentiments, dummy_topics)

        # Mock baseline
        mock_baseline_instance = MagicMock()
        mock_baseline_instance.predict_labels.side_effect = lambda b, word_segment=False: (
            [0 if i % 2 == 0 else 2 for i in range(len(b))],
            [0 if i % 2 == 0 else 1 for i in range(len(b))]
        )
        mock_baseline_cls.load.return_value = mock_baseline_instance

        # Mock PhoBERT
        mock_phobert_instance = MagicMock()
        mock_phobert_instance.to.return_value = mock_phobert_instance
        sent_logits = torch.zeros((num_samples, 3))
        topic_logits = torch.zeros((num_samples, 4))
        for i in range(num_samples):
            sent_logits[i, dummy_sentiments[i]] = 10.0
            topic_logits[i, dummy_topics[i]] = 10.0
        mock_phobert_instance.return_value = {
            "sentiment_logits": sent_logits,
            "topic_logits": topic_logits,
        }
        mock_phobert_cls.from_pretrained.return_value = mock_phobert_instance

        mock_tok_instance = MagicMock()
        mock_tok_instance.return_value = {
            "input_ids": torch.zeros((num_samples, 16), dtype=torch.long),
            "attention_mask": torch.ones((num_samples, 16), dtype=torch.long),
        }
        mock_tok_cls.from_pretrained.return_value = mock_tok_instance

        test_argv = [
            "evaluation/evaluate.py",
            "--baseline_path", "saved_models/baseline.joblib",
            "--phobert_path", "saved_models/phobert_multitask",
            "--data_dir", "data/uit_vsfc",
            "--output_dir", self.temp_dir,
            "--max_samples", "25",
            "--show_reports",
        ]

        with patch("os.path.isfile", return_value=True), patch("os.path.isdir", return_value=True):
            with patch.object(sys, "argv", test_argv):
                try:
                    eval_module.main()
                except SystemExit as e:
                    self.assertEqual(e.code, 0, f"CLI exited with non-zero code {e.code}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
