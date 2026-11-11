"""
tests/test_baseline.py
Comprehensive Unit Test Suite for Classical Baseline Model (Milestone 2).

Verifies:
- Model initialization and default hyperparameters
- Unfitted model exception handling
- Fitting on synthetic dummy data and UIT-VSFC validation data
- Strict conformance to PROJECT.md § Interface Contracts (Schema & Python native types)
- Probability calibration and summation to 1.0
- Persistence and Save/Load idempotence
- Robust edge case handling (empty strings, whitespace, empty lists, single string vs batch)

Execution:
    python -m unittest tests/test_baseline.py -v
"""

import math
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from models.baseline_model import (
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
    SENTIMENT_TO_ID,
    TOPIC_LABELS,
    TOPIC_MAP,
    TOPIC_TO_ID,
    VietnameseFeedbackBaseline,
)


class TestBaselineModelInitialization(unittest.TestCase):
    """Verifies baseline model instantiation, parameter defaults, and unfitted state protection."""

    def test_default_initialization(self):
        model = VietnameseFeedbackBaseline()
        self.assertFalse(model.is_fitted)
        self.assertEqual(model.ngram_range, (1, 2))
        self.assertEqual(model.min_df, 2)
        self.assertEqual(model.max_df, 0.9)
        self.assertEqual(model.max_features, 10000)
        self.assertTrue(model.sublinear_tf)
        self.assertEqual(model.C, 1.0)
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.class_weight, "balanced")
        self.assertEqual(model.random_state, 42)
        self.assertEqual(model.sentiment_labels, ["negative", "neutral", "positive"])
        self.assertEqual(model.topic_labels, ["lecturer", "training_program", "facility", "others"])

    def test_unfitted_model_raises_runtime_error(self):
        model = VietnameseFeedbackBaseline()
        with self.assertRaises(RuntimeError):
            model.predict("Thầy dạy rất nhiệt tình")
        with self.assertRaises(RuntimeError):
            model.predict_proba("Thầy dạy rất nhiệt tình")
        with self.assertRaises(RuntimeError):
            model.predict_labels("Thầy dạy rất nhiệt tình")
        with self.assertRaises(RuntimeError):
            model.save("saved_models/dummy.joblib")


class TestBaselineFittingAndSchema(unittest.TestCase):
    """Verifies fitting logic, label validation, and unified prediction dictionary schema."""

    @classmethod
    def setUpClass(cls):
        # Synthetic dataset covering all 3 sentiment classes and all 4 topic classes
        cls.dummy_texts = [
            "Thầy dạy rất nhiệt tình và chu đáo",       # pos (2), lec (0)
            "Giảng viên giải thích bài tập dễ hiểu",    # pos (2), lec (0)
            "Cô giảng bài khó hiểu quá buồn",           # neg (0), lec (0)
            "Nội dung chương trình đào tạo hợp lý",     # pos (2), prog (1)
            "Khung chương trình hơi nặng và lý thuyết", # neu (1), prog (1)
            "Phòng học máy lạnh hỏng nóng quá",         # neg (0), fac (2)
            "Bàn ghế phòng thực hành rất hiện đại",     # pos (2), fac (2)
            "Mong nhà trường cải thiện bãi giữ xe",     # neu (1), others (3)
            "Học phí đóng trễ bị phạt thêm tiền",       # neg (0), others (3)
            "Hoạt động ngoại khóa rất bổ ích",          # pos (2), others (3)
        ]
        cls.dummy_sentiments = [2, 2, 0, 2, 1, 0, 2, 1, 0, 2]
        cls.dummy_topics = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3]

        cls.model = VietnameseFeedbackBaseline(min_df=1, max_features=500)
        cls.model.fit(
            cls.dummy_texts,
            cls.dummy_sentiments,
            cls.dummy_topics,
            word_segment=True,
        )

    def test_mismatched_dataset_lengths_raises_value_error(self):
        model = VietnameseFeedbackBaseline(min_df=1)
        with self.assertRaises(ValueError):
            model.fit(["Text 1", "Text 2"], [1], [0])
        with self.assertRaises(ValueError):
            model.fit(["Text 1"], [1], [0, 1])

    def test_single_string_prediction_schema(self):
        pred = self.model.predict("Thầy giáo giảng bài rất hay")

        # Must return dict for single string
        self.assertIsInstance(pred, dict)
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)

        # Check sentiment schema
        sent = pred["sentiment"]
        self.assertIn("label", sent)
        self.assertIn("label_id", sent)
        self.assertIn("confidence", sent)
        self.assertIn("distribution", sent)

        self.assertIsInstance(sent["label"], str)
        self.assertIn(sent["label"], SENTIMENT_LABELS)
        self.assertIsInstance(sent["label_id"], int)
        self.assertEqual(type(sent["label_id"]), int)
        self.assertIn(sent["label_id"], [0, 1, 2])
        self.assertIsInstance(sent["confidence"], float)
        self.assertEqual(type(sent["confidence"]), float)
        self.assertGreaterEqual(sent["confidence"], 0.0)
        self.assertLessEqual(sent["confidence"], 1.0)

        # Distribution schema
        sent_dist = sent["distribution"]
        self.assertIsInstance(sent_dist, dict)
        self.assertEqual(set(sent_dist.keys()), {"negative", "neutral", "positive"})
        for k, v in sent_dist.items():
            self.assertEqual(type(v), float)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # Check topic schema
        topic = pred["topic"]
        self.assertIn("label", topic)
        self.assertIn("label_id", topic)
        self.assertIn("confidence", topic)
        self.assertIn("distribution", topic)

        self.assertIsInstance(topic["label"], str)
        self.assertIn(topic["label"], TOPIC_LABELS)
        self.assertIsInstance(topic["label_id"], int)
        self.assertEqual(type(topic["label_id"]), int)
        self.assertIn(topic["label_id"], [0, 1, 2, 3])
        self.assertIsInstance(topic["confidence"], float)
        self.assertEqual(type(topic["confidence"]), float)
        self.assertGreaterEqual(topic["confidence"], 0.0)
        self.assertLessEqual(topic["confidence"], 1.0)

        # Distribution schema
        topic_dist = topic["distribution"]
        self.assertIsInstance(topic_dist, dict)
        self.assertEqual(set(topic_dist.keys()), {"lecturer", "training_program", "facility", "others"})
        for k, v in topic_dist.items():
            self.assertEqual(type(v), float)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # Confidence must match distribution of predicted label
        self.assertAlmostEqual(sent["confidence"], sent_dist[sent["label"]], places=6)
        self.assertAlmostEqual(topic["confidence"], topic_dist[topic["label"]], places=6)

    def test_batch_prediction_schema(self):
        batch = [
            "Giảng viên dạy tốt",
            "Phòng thực hành hỏng điều hòa",
            "Chương trình học hợp lý",
        ]
        preds = self.model.predict(batch)

        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), len(batch))
        for p in preds:
            self.assertIsInstance(p, dict)
            self.assertIn("sentiment", p)
            self.assertIn("topic", p)
            self.assertEqual(type(p["sentiment"]["confidence"]), float)
            self.assertEqual(type(p["sentiment"]["label_id"]), int)
            self.assertEqual(type(p["topic"]["confidence"]), float)
            self.assertEqual(type(p["topic"]["label_id"]), int)

    def test_probability_summation_to_one(self):
        queries = [
            "Thầy giáo dạy rất nhiệt tình",
            "Cơ sở vật chất trường cần nâng cấp",
            "Khóa học quá khó hiểu",
        ]
        for query in queries:
            pred = self.model.predict(query)
            sent_dist_sum = sum(pred["sentiment"]["distribution"].values())
            topic_dist_sum = sum(pred["topic"]["distribution"].values())

            self.assertTrue(
                math.isclose(sent_dist_sum, 1.0, rel_tol=1e-5),
                f"Sentiment probability sum {sent_dist_sum} != 1.0",
            )
            self.assertTrue(
                math.isclose(topic_dist_sum, 1.0, rel_tol=1e-5),
                f"Topic probability sum {topic_dist_sum} != 1.0",
            )

    def test_predict_proba_and_predict_labels_vectorized(self):
        texts = ["Giảng viên tận tâm", "Máy chiếu phòng B201 bị mờ"]
        sent_probs, topic_probs = self.model.predict_proba(texts)

        self.assertIsInstance(sent_probs, np.ndarray)
        self.assertIsInstance(topic_probs, np.ndarray)
        self.assertEqual(sent_probs.shape, (2, 3))
        self.assertEqual(topic_probs.shape, (2, 4))

        # Check row sums
        np.testing.assert_allclose(sent_probs.sum(axis=1), np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(topic_probs.sum(axis=1), np.ones(2), rtol=1e-5)

        sent_ids, topic_ids = self.model.predict_labels(texts)
        self.assertIsInstance(sent_ids, list)
        self.assertIsInstance(topic_ids, list)
        self.assertEqual(len(sent_ids), 2)
        self.assertEqual(len(topic_ids), 2)
        self.assertEqual(sent_ids, np.argmax(sent_probs, axis=1).tolist())
        self.assertEqual(topic_ids, np.argmax(topic_probs, axis=1).tolist())


class TestBaselineEdgeCases(unittest.TestCase):
    """Verifies edge case handling: empty list, empty string, whitespace, emojis, punctuation."""

    @classmethod
    def setUpClass(cls):
        cls.model = VietnameseFeedbackBaseline(min_df=1, max_features=500)
        cls.model.fit(
            [
                "Thầy dạy tốt",
                "Phòng học tệ",
                "Nội dung bình thường",
                "Khóa học ổn",
            ],
            [2, 0, 1, 2],
            [0, 2, 1, 3],
        )

    def test_empty_list_input(self):
        pred = self.model.predict([])
        self.assertEqual(pred, [])

        sent_probs, topic_probs = self.model.predict_proba([])
        self.assertEqual(sent_probs.shape, (0, 3))
        self.assertEqual(topic_probs.shape, (0, 4))

        s_labels, t_labels = self.model.predict_labels([])
        self.assertEqual(s_labels, [])
        self.assertEqual(t_labels, [])

    def test_empty_string_input(self):
        pred = self.model.predict("")
        self.assertIsInstance(pred, dict)
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)
        self.assertTrue(math.isclose(sum(pred["sentiment"]["distribution"].values()), 1.0, rel_tol=1e-5))
        self.assertTrue(math.isclose(sum(pred["topic"]["distribution"].values()), 1.0, rel_tol=1e-5))

    def test_whitespace_string_input(self):
        pred = self.model.predict("   \n\t  ")
        self.assertIsInstance(pred, dict)
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)

    def test_single_element_list_returns_list(self):
        pred = self.model.predict(["Thầy dạy tốt"])
        self.assertIsInstance(pred, list)
        self.assertEqual(len(pred), 1)
        self.assertIsInstance(pred[0], dict)

    def test_special_characters_and_emojis(self):
        pred = self.model.predict("Giảng viên tuyệt vời quá! ❤️❤️❤️ 10/10 ⭐⭐⭐")
        self.assertIsInstance(pred, dict)
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)

    def test_none_input_handled_gracefully(self):
        # Passing None wrapped in list
        preds = self.model.predict([None, "Thầy dạy tốt"])
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 2)


class TestBaselinePersistence(unittest.TestCase):
    """Verifies save and load idempotence, directory auto-creation, and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = os.path.join(self.temp_dir, "nested", "baseline.joblib")

        self.model = VietnameseFeedbackBaseline(min_df=1, max_features=500)
        self.model.fit(
            [
                "Thầy dạy rất tận tình và nhiệt huyết",
                "Phòng học nóng, máy lạnh hư hỏng",
                "Chương trình học cân đối",
                "Bãi đỗ xe quá tải",
            ],
            [2, 0, 1, 0],
            [0, 2, 1, 3],
        )

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_creates_directory_and_file(self):
        saved_file = self.model.save(self.save_path)
        self.assertTrue(os.path.isfile(saved_file))
        self.assertEqual(saved_file, os.path.abspath(self.save_path))

    def test_save_and_load_idempotence(self):
        self.model.save(self.save_path)
        loaded_model = VietnameseFeedbackBaseline.load(self.save_path)

        self.assertIsInstance(loaded_model, VietnameseFeedbackBaseline)
        self.assertTrue(loaded_model.is_fitted)

        test_inputs = [
            "Thầy dạy rất nhiệt tình",
            "Phòng máy lạnh bị hỏng",
            "Chương trình đào tạo tốt",
        ]
        original_preds = self.model.predict(test_inputs)
        loaded_preds = loaded_model.predict(test_inputs)

        self.assertEqual(original_preds, loaded_preds)

        orig_s_prob, orig_t_prob = self.model.predict_proba(test_inputs)
        loaded_s_prob, loaded_t_prob = loaded_model.predict_proba(test_inputs)

        np.testing.assert_allclose(orig_s_prob, loaded_s_prob, rtol=1e-5)
        np.testing.assert_allclose(orig_t_prob, loaded_t_prob, rtol=1e-5)

    def test_load_nonexistent_file_raises_not_found(self):
        bogus_path = os.path.join(self.temp_dir, "nonexistent.joblib")
        with self.assertRaises(FileNotFoundError):
            VietnameseFeedbackBaseline.load(bogus_path)


class TestBaselineRealDataIntegration(unittest.TestCase):
    """Verifies fitting on real UIT-VSFC validation split samples."""

    @classmethod
    def setUpClass(cls):
        from data.dataset import load_vsfc_split

        # Load small slice of real data for fast verification
        cls.texts, cls.sents, cls.topics = load_vsfc_split("validation", word_segment=True)
        # Use first 300 samples for fast unittest fitting
        cls.sample_texts = cls.texts[:300]
        cls.sample_sents = cls.sents[:300]
        cls.sample_topics = cls.topics[:300]

    def test_fit_and_predict_on_real_samples(self):
        model = VietnameseFeedbackBaseline(min_df=2, max_features=1000)
        model.fit(
            self.sample_texts,
            self.sample_sents,
            self.sample_topics,
            word_segment=False,  # Already pre-segmented by load_vsfc_split
        )

        test_sent = "thầy_giáo dạy rất hay và nhiệt_tình"
        pred = model.predict(test_sent, word_segment=False)
        self.assertIsInstance(pred, dict)
        self.assertEqual(pred["sentiment"]["label"], "positive")
        self.assertEqual(pred["topic"]["label"], "lecturer")

class TestSavedBaselineArtifact(unittest.TestCase):
    """Verifies that the serialized saved_models/baseline.joblib artifact is intact and operational."""

    @classmethod
    def setUpClass(cls):
        cls.artifact_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")

    def test_artifact_exists_and_size_within_budget(self):
        self.assertTrue(
            os.path.isfile(self.artifact_path),
            f"Expected baseline artifact at {self.artifact_path} does not exist.",
        )
        file_size_bytes = os.path.getsize(self.artifact_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        # Ensure file size is under 2.0 MB (and well below 8.0 MB budget)
        self.assertLess(
            file_size_mb,
            2.0,
            f"Artifact size {file_size_mb:.2f} MB exceeds 2.0 MB threshold.",
        )
        self.assertGreater(
            file_size_bytes,
            100000,
            f"Artifact size {file_size_bytes} bytes is suspiciously small.",
        )

    def test_loaded_artifact_inference_and_schema(self):
        if not os.path.isfile(self.artifact_path):
            self.skipTest(f"Artifact {self.artifact_path} not found.")

        model = VietnameseFeedbackBaseline.load(self.artifact_path)
        self.assertTrue(model.is_fitted)

        # Single string prediction
        sample = "Thầy cô giảng dạy rất tận tình và chu đáo"
        pred = model.predict(sample)
        self.assertIsInstance(pred, dict)
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)
        self.assertEqual(pred["sentiment"]["label"], "positive")
        self.assertEqual(pred["topic"]["label"], "lecturer")
        self.assertEqual(type(pred["sentiment"]["confidence"]), float)
        self.assertEqual(type(pred["sentiment"]["label_id"]), int)
        self.assertEqual(type(pred["topic"]["confidence"]), float)
        self.assertEqual(type(pred["topic"]["label_id"]), int)

        # Facility sample
        fac_sample = "Cơ sở vật chất phòng máy quá cũ kỹ, máy lạnh thường xuyên hư"
        fac_pred = model.predict(fac_sample)
        self.assertEqual(fac_pred["sentiment"]["label"], "negative")
        self.assertEqual(fac_pred["topic"]["label"], "facility")


if __name__ == "__main__":
    unittest.main()

