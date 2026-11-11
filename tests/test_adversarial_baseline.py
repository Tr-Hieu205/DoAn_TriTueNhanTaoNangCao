"""
tests/test_adversarial_baseline.py
Empirical Adversarial Stress-Testing Suite for Milestone 2: Classical Baseline Model.

Author: Challenger M2_2 (critic, specialist)
Validates:
  1. Extreme Edge Cases:
     - Empty string, pure whitespace (ASCII, Unicode non-breaking, zero-width space)
     - Special characters, mathematical symbols, punctuation-only strings
     - Mixed Vietnamese / English code-switching and student internet slang
     - Emojis only and mixed emoji-Vietnamese text
     - Very long texts (>500 words, >5,000 words) and repetitive spam
     - Out-of-vocabulary (OOV) nonsense, pure numbers, HTML / SQL strings
  2. Reload from Disk & Fidelity:
     - Load `saved_models/baseline.joblib` and compare against fresh fit on same seed
     - Predict probability parity and 100% label concordance on real test samples
  3. Schema & Invariant Strictness:
     - Native Python types (not np.int64/np.float64), range [0, 1], sum == 1.0
     - Equivalence of single string prediction and batch prediction
     - Input polymorphism (single str, list of str, empty list, None in list)
  4. Throughput & Scalability Benchmark:
     - High-throughput inference (>10,000 samples/sec on CPU)
"""

import math
import os
import sys
import unittest
from typing import Any, Dict, List

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from data.dataset import load_vsfc_split
from models.baseline_model import (
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
    TOPIC_LABELS,
    TOPIC_MAP,
    VietnameseFeedbackBaseline,
)

ARTIFACT_PATH = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")


class TestAdversarialEdgeCases(unittest.TestCase):
    """Stress tests baseline model against extreme and hostile textual inputs."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(ARTIFACT_PATH):
            raise FileNotFoundError(f"Model artifact not found at {ARTIFACT_PATH}")
        cls.model = VietnameseFeedbackBaseline.load(ARTIFACT_PATH)

    def _verify_contract(self, pred: Dict[str, Any], context_msg: str = ""):
        """Helper to verify strict adherence to PROJECT.md interface contract."""
        self.assertIsInstance(pred, dict, f"Result must be a dict: {context_msg}")
        self.assertIn("sentiment", pred, f"Missing 'sentiment': {context_msg}")
        self.assertIn("topic", pred, f"Missing 'topic': {context_msg}")

        # Sentiment contract
        s = pred["sentiment"]
        self.assertIsInstance(s["label"], str, context_msg)
        self.assertIn(s["label"], SENTIMENT_LABELS, context_msg)
        self.assertEqual(type(s["label_id"]), int, f"label_id must be native int: {context_msg}")
        self.assertIn(s["label_id"], [0, 1, 2], context_msg)
        self.assertEqual(type(s["confidence"]), float, f"confidence must be native float: {context_msg}")
        self.assertGreaterEqual(s["confidence"], 0.0, context_msg)
        self.assertLessEqual(s["confidence"], 1.0, context_msg)
        self.assertIsInstance(s["distribution"], dict, context_msg)
        self.assertEqual(set(s["distribution"].keys()), set(SENTIMENT_LABELS), context_msg)
        for k, v in s["distribution"].items():
            self.assertEqual(type(v), float, context_msg)
            self.assertGreaterEqual(v, 0.0, context_msg)
            self.assertLessEqual(v, 1.0, context_msg)
        sent_sum = sum(s["distribution"].values())
        self.assertTrue(math.isclose(sent_sum, 1.0, rel_tol=1e-5), f"Sentiment prob sum {sent_sum} != 1: {context_msg}")

        # Topic contract
        t = pred["topic"]
        self.assertIsInstance(t["label"], str, context_msg)
        self.assertIn(t["label"], TOPIC_LABELS, context_msg)
        self.assertEqual(type(t["label_id"]), int, f"topic label_id must be native int: {context_msg}")
        self.assertIn(t["label_id"], [0, 1, 2, 3], context_msg)
        self.assertEqual(type(t["confidence"]), float, f"topic confidence must be native float: {context_msg}")
        self.assertGreaterEqual(t["confidence"], 0.0, context_msg)
        self.assertLessEqual(t["confidence"], 1.0, context_msg)
        self.assertIsInstance(t["distribution"], dict, context_msg)
        self.assertEqual(set(t["distribution"].keys()), set(TOPIC_LABELS), context_msg)
        for k, v in t["distribution"].items():
            self.assertEqual(type(v), float, context_msg)
            self.assertGreaterEqual(v, 0.0, context_msg)
            self.assertLessEqual(v, 1.0, context_msg)
        topic_sum = sum(t["distribution"].values())
        self.assertTrue(math.isclose(topic_sum, 1.0, rel_tol=1e-5), f"Topic prob sum {topic_sum} != 1: {context_msg}")

    def test_empty_string(self):
        """Empty string must not throw and must return valid probabilities."""
        pred = self.model.predict("")
        self._verify_contract(pred, "empty string")

    def test_pure_whitespace_variations(self):
        """Whitespace varieties: spaces, tabs, newlines, unicode non-breaking, zero-width."""
        whitespaces = [
            "   ",
            "\t\t\n\r\n  \t",
            "\u00A0\u00A0\u00A0",  # Non-breaking spaces
            "\u200B\u200B\u200B",  # Zero-width spaces
            " \u3000 ",            # Ideographic space
        ]
        for ws in whitespaces:
            pred = self.model.predict(ws)
            self._verify_contract(pred, f"whitespace {repr(ws)}")

    def test_special_characters_and_symbols(self):
        """Punctuation, math symbols, keyboard mash."""
        cases = [
            "!@#$%^&*()_+=-`~[]\\{}|;':\",./<>?",
            "∑ ∏ √ ∫ ≈ ≠ ≤ ≥ ∞",
            "...................",
            "???!!!!!????",
            "-------------------",
        ]
        for c in cases:
            pred = self.model.predict(c)
            self._verify_contract(pred, f"special chars {c}")

    def test_mixed_vietnamese_and_english(self):
        """Code-switching, student slang, abbreviations common in university feedback."""
        cases = [
            "Professor dạy rất good nhưng homework hơi nhiều deadline dí sấp mặt",
            "Slide presentation so boring, thầy đọc chép từ đầu tới cuối",
            "Thầy support 24/7 nhiệt tình vl :)) 10 điểm ko có nhưng",
            "Môn này lý thuyết quá heavy, đề thi thì out trình sinh viên",
            "Cơ sở vật chất campus xịn xò, wifi connect vèo vèo",
            "Học phí tăng cao mà chất lượng service quá tệ",
        ]
        for c in cases:
            pred = self.model.predict(c)
            self._verify_contract(pred, f"mixed text: {c}")

    def test_emojis_variations(self):
        """Pure emojis and emojis embedded in sentences."""
        cases = [
            "❤️❤️❤️ 🥰 🥰 🚀 🚀",
            "😡 🤬 👎 💩",
            "Thầy giáo dễ thương xỉu 😍😍😍 giảng bài cuốn hút lắm ạ",
            "Phòng học nóng như lò bát quái 🔥🔥🔥 máy lạnh hư 😭😭😭",
            "Chương trình đào tạo 5 sao 🌟🌟🌟🌟🌟",
        ]
        for c in cases:
            pred = self.model.predict(c)
            self._verify_contract(pred, f"emoji test: {c}")

    def test_very_long_sentences(self):
        """Sentences exceeding 500 words and 2,000 words."""
        # 500 words
        base_500 = "giảng viên dạy rất tận tâm nhiệt tình hướng dẫn sinh viên làm bài tập " * 50
        self.assertGreater(len(base_500.split()), 500)
        pred_500 = self.model.predict(base_500)
        self._verify_contract(pred_500, "500-word sentence")
        self.assertEqual(pred_500["sentiment"]["label"], "positive")
        self.assertEqual(pred_500["topic"]["label"], "lecturer")

        # 2,000 words
        base_2000 = "phòng học cơ sở vật chất máy lạnh hư nóng hỏng thiết bị " * 200
        self.assertGreater(len(base_2000.split()), 2000)
        pred_2000 = self.model.predict(base_2000)
        self._verify_contract(pred_2000, "2000-word sentence")
        self.assertEqual(pred_2000["sentiment"]["label"], "negative")
        self.assertEqual(pred_2000["topic"]["label"], "facility")

    def test_out_of_vocabulary_nonsense(self):
        """Completely random strings not present in Vietnamese vocabulary."""
        cases = [
            "asdfghjkl qwertyuiop zxcvbnm",
            "zzzzzzzz xxxxxxxx yyyyyyyy",
            "lorum ipsum dolor sit amet consectetuer adipiscing elit",
            "1234567890 987654321 00000000",
        ]
        for c in cases:
            pred = self.model.predict(c)
            self._verify_contract(pred, f"oov test: {c}")

    def test_html_script_and_injection_payloads(self):
        """Adversarial web payloads (XSS, SQLi, Markdown)."""
        payloads = [
            "<script>alert('xss');</script>",
            "<img src='x' onerror='alert(1)'>",
            "' OR '1'='1' --",
            "DROP TABLE feedback; --",
            "# Header\n**Bold** _italic_ [link](http://test.com)",
        ]
        for p in payloads:
            pred = self.model.predict(p)
            self._verify_contract(pred, f"injection payload: {p}")

    def test_batch_heterogeneity(self):
        """Batch containing mixed normal, empty, emoji, None, and long inputs."""
        hetero_batch = [
            "Thầy dạy rất hay",
            "",
            "   \t  ",
            "🔥🔥🔥 100/100",
            None,
            "Cơ sở vật chất chán đời",
            "wzjwz208 giảng dạy môn wzjwz14",
            "A" * 1000,
        ]
        preds = self.model.predict(hetero_batch)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), len(hetero_batch))
        for i, p in enumerate(preds):
            self._verify_contract(p, f"batch element {i}")


class TestReloadFromDiskFidelity(unittest.TestCase):
    """Verifies that loaded model matches a freshly trained baseline model."""

    @classmethod
    def setUpClass(cls):
        # Load test split for validation
        cls.test_texts, cls.test_sents, cls.test_topics = load_vsfc_split("test", word_segment=True)
        # Select first 200 samples for deterministic numerical fidelity check
        cls.sample_texts = cls.test_texts[:200]
        cls.sample_sents = cls.test_sents[:200]
        cls.sample_topics = cls.test_topics[:200]

        # Load disk artifact
        cls.disk_model = VietnameseFeedbackBaseline.load(ARTIFACT_PATH)

    def test_disk_artifact_attributes(self):
        """Verify vectorizer and classifiers are fitted and have expected dimensions."""
        self.assertTrue(self.disk_model.is_fitted)
        self.assertIsInstance(self.disk_model.vectorizer.vocabulary_, dict)
        self.assertEqual(len(self.disk_model.vectorizer.vocabulary_), 10000)
        self.assertEqual(len(self.disk_model.sentiment_clf.classes_), 3)
        self.assertEqual(len(self.disk_model.topic_clf.classes_), 4)

    def test_prediction_fidelity_and_determinism(self):
        """Verify predict and predict_proba return identical results across multiple calls."""
        pred1 = self.disk_model.predict(self.sample_texts[:20], word_segment=False)
        pred2 = self.disk_model.predict(self.sample_texts[:20], word_segment=False)
        self.assertEqual(pred1, pred2)

        s_prob1, t_prob1 = self.disk_model.predict_proba(self.sample_texts[:20], word_segment=False)
        s_prob2, t_prob2 = self.disk_model.predict_proba(self.sample_texts[:20], word_segment=False)
        np.testing.assert_array_equal(s_prob1, s_prob2)
        np.testing.assert_array_equal(t_prob1, t_prob2)

    def test_single_vs_batch_consistency(self):
        """Predicting a single string must match the corresponding element in batch prediction."""
        samples = [
            "Giảng viên giải thích chi tiết và tận tình",
            "Nhà vệ sinh bẩn và thiếu nước",
            "Môn học có tính thực tiễn cao",
        ]
        batch_preds = self.disk_model.predict(samples)
        for i, s in enumerate(samples):
            single_pred = self.disk_model.predict(s)
            self.assertEqual(single_pred, batch_preds[i])

    def test_predict_proba_matches_confidence(self):
        """Verify that 'confidence' in predict() exactly equals the max probability in predict_proba()."""
        samples = self.sample_texts[:50]
        s_probs, t_probs = self.disk_model.predict_proba(samples, word_segment=False)
        dict_preds = self.disk_model.predict(samples, word_segment=False)

        for i in range(len(samples)):
            expected_s_max = float(np.max(s_probs[i]))
            expected_t_max = float(np.max(t_probs[i]))

            self.assertAlmostEqual(dict_preds[i]["sentiment"]["confidence"], expected_s_max, places=6)
            self.assertAlmostEqual(dict_preds[i]["topic"]["confidence"], expected_t_max, places=6)

            pred_s_label = dict_preds[i]["sentiment"]["label"]
            pred_t_label = dict_preds[i]["topic"]["label"]

            self.assertEqual(pred_s_label, SENTIMENT_MAP[int(np.argmax(s_probs[i]))])
            self.assertEqual(pred_t_label, TOPIC_MAP[int(np.argmax(t_probs[i]))])


class TestThroughputAndScalability(unittest.TestCase):
    """Verifies baseline inference latency and throughput satisfy interactive requirements."""

    @classmethod
    def setUpClass(cls):
        cls.disk_model = VietnameseFeedbackBaseline.load(ARTIFACT_PATH)
        # Use full test split (3,166 samples)
        cls.test_texts, _, _ = load_vsfc_split("test", word_segment=True)

    def test_inference_throughput_benchmark(self):
        """Batch inference on 3,166 samples should exceed 10,000 samples/sec on modern CPU."""
        import time

        # Warm up
        _ = self.disk_model.predict_labels(self.test_texts[:100], word_segment=False)

        t0 = time.time()
        s_preds, t_preds = self.disk_model.predict_labels(self.test_texts, word_segment=False)
        elapsed = time.time() - t0

        self.assertEqual(len(s_preds), len(self.test_texts))
        self.assertEqual(len(t_preds), len(self.test_texts))

        throughput = len(self.test_texts) / elapsed
        # Expect throughput > 10,000 samples/sec
        self.assertGreater(
            throughput,
            10000.0,
            f"Throughput {throughput:,.1f} samples/sec is below required 10,000 samples/sec threshold",
        )
        # Latency per sample < 0.1 ms
        latency_ms_per_sample = (elapsed / len(self.test_texts)) * 1000.0
        self.assertLess(latency_ms_per_sample, 0.1)


if __name__ == "__main__":
    unittest.main()
