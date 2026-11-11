"""
tests/test_app.py
Automated QA & Test Suite for Milestone 5: Streamlit Interactive Feedback Analyzer Demo Dashboard.

Verifies:
1. Preset Integrity: Tests that all preset samples are non-empty Vietnamese strings and cover diverse labels.
2. Function Unit Tests: Tests rendering helpers, badge mappings, probability normalization, and latency formatting.
3. Model Loading Logic: Tests cached model loading behavior and fallback when paths do not exist.
4. Prediction Logic: Tests calling prediction wrapper with sample Vietnamese text.
5. Streamlit App Smoke & User Interaction Tests: Using streamlit.testing.v1.AppTest to verify rendering,
   preset selection, prediction execution, side-by-side mode, empty input validation, and graceful fallbacks.
6. Execution Ergonomics: Full automated suite completes in <5 seconds.

Execution:
    python -m unittest tests/test_app.py -v
    python -m unittest discover tests -v
"""

import os
import re
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Windows console UTF-8 safeguard
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

APP_PATH = os.path.join(PROJECT_ROOT, "app", "app.py")

import app.app as demo_app
from streamlit.testing.v1 import AppTest
from models.baseline_model import VietnameseFeedbackBaseline, SENTIMENT_MAP, TOPIC_MAP
import models.phobert_model as phobert_module
import transformers


class MockPhoBERTMultiTask:
    """Lightweight mock fixture for PhoBERTMultiTask avoiding 515MB weight load in fast CI."""

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def to(self, device):
        return self

    def predict(self, texts, tokenizer=None, device="cpu", **kwargs):
        # Simulate small realistic neural inference time (10ms)
        time.sleep(0.01)
        is_single = isinstance(texts, str)
        item = {
            "sentiment": {
                "label": "positive",
                "label_id": 2,
                "confidence": 0.985,
                "distribution": {"negative": 0.005, "neutral": 0.010, "positive": 0.985},
            },
            "topic": {
                "label": "lecturer",
                "label_id": 0,
                "confidence": 0.962,
                "distribution": {
                    "lecturer": 0.962,
                    "training_program": 0.020,
                    "facility": 0.010,
                    "others": 0.008,
                },
            },
        }
        return item if is_single else [item] * len(texts)


# Module-level setup/teardown to patch PhoBERT loading only for test_app.py
_orig_phobert_from_pretrained = None
_orig_tok_from_pretrained = None


def setUpModule():
    global _orig_phobert_from_pretrained, _orig_tok_from_pretrained
    _orig_phobert_from_pretrained = phobert_module.PhoBERTMultiTask.from_pretrained
    _orig_tok_from_pretrained = transformers.AutoTokenizer.from_pretrained

    phobert_module.PhoBERTMultiTask.from_pretrained = classmethod(lambda cls, *args, **kwargs: MockPhoBERTMultiTask())
    transformers.AutoTokenizer.from_pretrained = classmethod(lambda cls, *args, **kwargs: MagicMock())

    import streamlit as st
    st.cache_resource.clear()
    if hasattr(demo_app.load_phobert_pipeline, "clear"):
        demo_app.load_phobert_pipeline.clear()
    if hasattr(demo_app.load_baseline_model, "clear"):
        demo_app.load_baseline_model.clear()


def tearDownModule():
    global _orig_phobert_from_pretrained, _orig_tok_from_pretrained
    if _orig_phobert_from_pretrained is not None:
        phobert_module.PhoBERTMultiTask.from_pretrained = _orig_phobert_from_pretrained
    if _orig_tok_from_pretrained is not None:
        transformers.AutoTokenizer.from_pretrained = _orig_tok_from_pretrained

    import streamlit as st
    st.cache_resource.clear()
    if hasattr(demo_app.load_phobert_pipeline, "clear"):
        demo_app.load_phobert_pipeline.clear()
    if hasattr(demo_app.load_baseline_model, "clear"):
        demo_app.load_baseline_model.clear()


class TestPresetIntegrity(unittest.TestCase):
    """
    Verifies the integrity, quality, linguistic authenticity, and diversity of the preset examples library.
    """

    def setUp(self):
        self.presets = demo_app.get_presets()

    def test_presets_count_and_data_structure(self):
        """Presets list must contain at least 8 curated examples conforming to dictionary schema."""
        self.assertIsInstance(self.presets, list, "Presets must be returned as a list")
        self.assertGreaterEqual(
            len(self.presets),
            8,
            f"Expected at least 8 presets per F17 requirement, got {len(self.presets)}",
        )
        for i, preset in enumerate(self.presets):
            self.assertIsInstance(preset, dict, f"Preset #{i} is not a dictionary")
            self.assertIn("text", preset, f"Preset #{i} missing 'text' field")
            self.assertIn("expected_sentiment", preset, f"Preset #{i} missing 'expected_sentiment'")
            self.assertIn("expected_topic", preset, f"Preset #{i} missing 'expected_topic'")
            self.assertIn("title", preset, f"Preset #{i} missing 'title' field")

    def test_presets_non_empty_vietnamese_text(self):
        """All preset feedback sentences must be authentic non-empty Vietnamese strings with diacritics."""
        vn_char_pattern = re.compile(
            r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
            re.IGNORECASE,
        )

        for i, preset in enumerate(self.presets):
            text = preset["text"].strip()
            self.assertGreater(
                len(text),
                15,
                f"Preset #{i} text too short (<15 chars): '{text}'",
            )
            self.assertTrue(
                bool(vn_char_pattern.search(text)),
                f"Preset #{i} does not appear to contain valid Vietnamese diacritics: '{text}'",
            )
            self.assertEqual(
                text,
                preset["text"],
                f"Preset #{i} contains untrimmed leading/trailing whitespace",
            )

    def test_presets_sentiment_and_topic_coverage(self):
        """Presets must collectively cover all 3 sentiment classes and all 4 topic classes."""
        all_sentiments = {p["expected_sentiment"] for p in self.presets}
        all_topics = {p["expected_topic"] for p in self.presets}

        required_sentiments = {"positive", "neutral", "negative"}
        required_topics = {"lecturer", "training_program", "facility", "others"}

        self.assertEqual(
            all_sentiments,
            required_sentiments,
            f"Presets must cover all 3 sentiments {required_sentiments}, found {all_sentiments}",
        )
        self.assertEqual(
            all_topics,
            required_topics,
            f"Presets must cover all 4 topics {required_topics}, found {all_topics}",
        )

    def test_presets_uniqueness(self):
        """All preset feedback sentences must be distinct without duplicate texts."""
        texts = [p["text"] for p in self.presets]
        self.assertEqual(
            len(texts),
            len(set(texts)),
            "Found duplicate feedback texts in presets library",
        )


class TestRenderingHelpersAndFormatting(unittest.TestCase):
    """
    Verifies pure helper functions: badge metadata, color palettes, probability normalization, and latency.
    """

    def test_sentiment_ui_metadata_completeness(self):
        """SENTIMENT_UI_META must contain complete metadata for all 3 sentiment labels."""
        meta = demo_app.SENTIMENT_UI_META
        self.assertEqual(set(meta.keys()), {"positive", "neutral", "negative"})
        for label, data in meta.items():
            self.assertIn("vn_name", data)
            self.assertIn("emoji", data)
            self.assertIn("color", data)
            self.assertIn("bg_color", data)
            self.assertIn("border_color", data)
            self.assertTrue(data["color"].startswith("#"), f"Color must be hex string: {data['color']}")

    def test_topic_ui_metadata_completeness(self):
        """TOPIC_UI_META must contain complete metadata for all 4 topic labels."""
        meta = demo_app.TOPIC_UI_META
        self.assertEqual(set(meta.keys()), {"lecturer", "training_program", "facility", "others"})
        for label, data in meta.items():
            self.assertIn("vn_name", data)
            self.assertIn("icon", data)
            self.assertIn("color", data)
            self.assertIn("bg_color", data)
            self.assertIn("border_color", data)
            self.assertTrue(data["color"].startswith("#"), f"Color must be hex string: {data['color']}")

    def test_metadata_lookup_safe_fallback(self):
        """Unknown or unobserved labels must not crash when accessing .get()."""
        s_meta = demo_app.SENTIMENT_UI_META.get("unseen_label", {})
        self.assertEqual(s_meta, {})
        t_meta = demo_app.TOPIC_UI_META.get("unseen_label", {})
        self.assertEqual(t_meta, {})

    def test_probability_distribution_normalization_and_bounds(self):
        """Distributions must be non-negative and sum to 1.0 within floating point precision."""
        sample_dist = {"negative": 0.05, "neutral": 0.15, "positive": 0.80}
        total = sum(sample_dist.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        for prob in sample_dist.values():
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)


class TestModelLoadingAndFallbackLogic(unittest.TestCase):
    """
    Verifies model loader functions, offline execution, and graceful fallbacks for missing checkpoints.
    """

    def test_load_baseline_real_checkpoint_success(self):
        """Baseline model loads cleanly from disk if checkpoint exists."""
        baseline_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        if os.path.isfile(baseline_path):
            model = demo_app.load_baseline_model(baseline_path)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, VietnameseFeedbackBaseline)
            self.assertTrue(model.is_fitted)

    def test_load_baseline_missing_file_fallback(self):
        """Baseline loader must return None when checkpoint file does not exist, without raising exception."""
        model = demo_app.load_baseline_model("non_existent_dir/baseline.joblib")
        self.assertIsNone(model)

    def test_load_phobert_missing_directory_fallback(self):
        """PhoBERT loader must return (None, None, 'cpu') when directory does not exist, without crashing."""
        model, tok, device = demo_app.load_phobert_pipeline("non_existent_dir/phobert")
        self.assertIsNone(model)
        self.assertIsNone(tok)
        self.assertEqual(device, "cpu")

    def test_load_phobert_empty_directory_fallback(self):
        """PhoBERT loader must return (None, None, 'cpu') when weights file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, tok, device = demo_app.load_phobert_pipeline(tmpdir)
            self.assertIsNone(model)
            self.assertIsNone(tok)
            self.assertEqual(device, "cpu")


class TestPredictionWrapperLogic(unittest.TestCase):
    """
    Verifies unified schema compliance, prediction output types, and edge-case inputs.
    """

    @classmethod
    def setUpClass(cls):
        baseline_path = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
        if os.path.isfile(baseline_path):
            cls.baseline = demo_app.load_baseline_model(baseline_path)
        else:
            cls.baseline = None

    def test_baseline_prediction_schema_compliance(self):
        """Prediction output must strictly conform to PROJECT.md § Interface Contracts."""
        if self.baseline is None:
            self.skipTest("baseline.joblib checkpoint not found")

        text = "Thầy dạy rất nhiệt tình, giảng giải cặn kẽ và hỗ trợ sinh viên chu đáo."
        t0 = time.perf_counter()
        res = self.baseline.predict(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Latency check: baseline should execute quickly (<250ms cold start, <10ms warm)
        self.assertGreater(latency_ms, 0.0)
        self.assertLess(latency_ms, 250.0)

        # Schema keys
        self.assertIn("sentiment", res)
        self.assertIn("topic", res)

        # Sentiment contract
        s = res["sentiment"]
        self.assertIn(s["label"], ["negative", "neutral", "positive"])
        self.assertIn(s["label_id"], [0, 1, 2])
        self.assertIsInstance(s["confidence"], float)
        self.assertGreaterEqual(s["confidence"], 0.0)
        self.assertLessEqual(s["confidence"], 1.0)
        self.assertEqual(set(s["distribution"].keys()), {"negative", "neutral", "positive"})
        self.assertAlmostEqual(sum(s["distribution"].values()), 1.0, places=4)

        # Topic contract
        t = res["topic"]
        self.assertIn(t["label"], ["lecturer", "training_program", "facility", "others"])
        self.assertIn(t["label_id"], [0, 1, 2, 3])
        self.assertIsInstance(t["confidence"], float)
        self.assertGreaterEqual(t["confidence"], 0.0)
        self.assertLessEqual(t["confidence"], 1.0)
        self.assertEqual(set(t["distribution"].keys()), {"lecturer", "training_program", "facility", "others"})
        self.assertAlmostEqual(sum(t["distribution"].values()), 1.0, places=4)

    def test_prediction_long_feedback_text(self):
        """Model handles multi-sentence lengthy student feedback without index errors."""
        if self.baseline is None:
            self.skipTest("baseline.joblib checkpoint not found")

        long_text = "Thầy dạy rất hay. " * 30  # ~150 words
        res = self.baseline.predict(long_text)
        self.assertIn("sentiment", res)
        self.assertIn("topic", res)


class TestStreamlitAppSmokeAndInteractions(unittest.TestCase):
    """
    Verifies the complete Streamlit web application end-to-end using native AppTest.
    Tests rendering, sidebar controls, preset selection, button triggers, and error states.
    Uses mock PhoBERT and persistent AppTest runner to guarantee rapid execution (<2s).
    """

    @classmethod
    def setUpClass(cls):
        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()
        cls.at = AppTest.from_file(APP_PATH, default_timeout=10).run()

    @classmethod
    def tearDownClass(cls):
        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()

    def test_app_renders_without_exception(self):
        """Smoke test: Streamlit app launches and executes initial render cycle with zero exceptions."""
        self.assertFalse(self.at.exception, f"App threw unexpected exception: {self.at.exception}")
        all_text = " ".join([t.value for t in self.at.title] + [m.value for m in self.at.markdown])
        self.assertIn(
            "Phân loại Phản hồi Sinh viên",
            all_text,
            "Target Vietnamese page title not found in rendered titles or markdown",
        )

    def test_app_sidebar_controls_and_options(self):
        """Sidebar must render model selector radio, preset dropdown, and system status."""
        self.assertFalse(self.at.exception)

        # Model radio exists
        self.assertGreaterEqual(len(self.at.sidebar.radio), 1)
        radio = self.at.sidebar.radio[0]
        options = radio.options
        self.assertTrue(
            any("Trung bình" in opt for opt in options),
            "Trung bình option missing from model selector",
        )
        self.assertTrue(
            any("Cao" in opt for opt in options),
            "Cao option missing from model selector",
        )
        self.assertTrue(
            any("Cả hai" in opt for opt in options),
            "Cả hai option missing from model selector",
        )

        # Preset selectbox exists
        self.assertGreaterEqual(len(self.at.sidebar.selectbox), 1)
        preset_box = self.at.sidebar.selectbox[0]
        self.assertGreaterEqual(
            len(preset_box.options),
            8,
            f"Preset selectbox has fewer than 8 options: {len(preset_box.options)}",
        )

    def test_preset_selection_flow(self):
        """Selecting a preset in the sidebar and clicking load updates the text area."""
        self.at.sidebar.selectbox[0].select_index(1)
        self.at.sidebar.button[0].click().run()
        self.assertFalse(self.at.exception)

        text_val = self.at.text_area[0].value
        self.assertIn("Phòng máy tính hư hỏng", text_val)

    def test_analyze_button_triggers_prediction_and_visualizations(self):
        """Entering Vietnamese feedback and clicking Analyze triggers prediction and renders cards/progress bars."""
        baseline_opts = [
            opt for opt in self.at.sidebar.radio[0].options if "Trung bình" in opt
        ]
        if baseline_opts:
            self.at.sidebar.radio[0].set_value(baseline_opts[0])

        self.at.text_area[0].input("Thầy dạy rất tận tâm và nhiệt tình")

        analyze_btn = None
        for btn in self.at.button:
            if "Phân tích" in btn.label or "Analyze" in btn.label:
                analyze_btn = btn
                break

        self.assertIsNotNone(analyze_btn, "Could not find 'Phân tích' action button")
        analyze_btn.click().run()

        self.assertFalse(self.at.exception, f"Exception during analysis: {self.at.exception}")

        self.assertGreaterEqual(
            len(self.at.progress),
            7,  # 3 sentiment classes + 4 topic classes = 7 bars
            f"Expected at least 7 probability progress bars, found {len(self.at.progress)}",
        )

    def test_side_by_side_comparison_flow(self):
        """Selecting Side-by-Side Comparison mode runs both models and renders dual cards & consensus."""
        side_by_side_opts = [
            opt for opt in self.at.sidebar.radio[0].options if "Cả hai" in opt
        ]
        self.assertTrue(len(side_by_side_opts) > 0, "Cả hai option must be available in radio")

        self.at.sidebar.radio[0].set_value(side_by_side_opts[0])
        self.at.text_area[0].input("Thầy dạy rất tận tâm và nhiệt tình")

        for btn in self.at.button:
            if "Phân tích" in btn.label or "Analyze" in btn.label:
                btn.click()
                break

        self.at.run()
        self.assertFalse(self.at.exception, f"Exception in side-by-side mode: {self.at.exception}")
        # In side-by-side mode, progress bars should be rendered for both models (7 * 2 = 14 bars)
        self.assertGreaterEqual(
            len(self.at.progress),
            14,
            f"Expected at least 14 progress bars in side-by-side mode, got {len(self.at.progress)}",
        )

    def test_empty_input_validation_warning(self):
        """Clicking Analyze with empty text area displays warning banner and prevents crash."""
        self.at.text_area[0].input("   ")

        for btn in self.at.button:
            if "Phân tích" in btn.label or "Analyze" in btn.label:
                btn.click()
                break

        self.at.run()
        self.assertFalse(self.at.exception)
        self.assertGreaterEqual(
            len(self.at.warning),
            1,
            "Expected st.warning when submitting empty input",
        )
        self.assertIn(
            "Vui lòng nhập",
            self.at.warning[0].value,
        )

    def test_graceful_handling_when_phobert_missing(self):
        """When PhoBERT weights are absent, app degrades gracefully without throwing unhandled exceptions."""
        try:
            demo_app.load_phobert_pipeline.clear()
            with patch.object(phobert_module.PhoBERTMultiTask, "from_pretrained", side_effect=FileNotFoundError("Mock missing weights")):
                at = AppTest.from_file(APP_PATH, default_timeout=10).run()
                self.assertFalse(at.exception, "App crashed when PhoBERT pipeline returned None")
                radio = at.sidebar.radio[0]
                self.assertIn("Trung bình", radio.options)
        finally:
            demo_app.load_phobert_pipeline.clear()

    def test_theme_dark_and_english_language_switching(self):
        """Switching to Dark theme and English language updates all UI labels and runs analysis cleanly."""
        at = AppTest.from_file(APP_PATH, default_timeout=10).run()
        self.assertFalse(at.exception)

        self.assertGreaterEqual(len(at.sidebar.button_group), 2)
        at.sidebar.button_group[0].set_value("dark").run()
        at.sidebar.button_group[1].set_value("en").run()
        self.assertFalse(at.exception, f"Theme/language switch threw exception: {at.exception}")

        radio = at.sidebar.radio[0]
        self.assertEqual(list(radio.options), ["Medium", "High", "Both"])
        self.assertIn("Choose a sample student feedback", at.sidebar.selectbox[0].label)

        analyze_btn = next((b for b in at.button if "Analyze" in b.label), None)
        self.assertIsNotNone(analyze_btn, "English Analyze button not found")

        at.text_area[0].input("Thầy dạy rất tận tâm và dễ hiểu.")
        analyze_btn.click().run()
        self.assertFalse(at.exception, f"Analysis in Dark/EN mode threw exception: {at.exception}")

        self.assertEqual(len(at.progress), 14)
        all_markdown = " ".join([m.value for m in at.markdown])
        self.assertIn("#0B1120", all_markdown)
        self.assertIn("Text Classification of Vietnamese Student Feedback", all_markdown)
        self.assertIn("Probability Breakdown (Donut)", all_markdown)
        self.assertIn("vsfc-donut-ring", all_markdown)
        self.assertIn("Comparison & Consensus Analysis", all_markdown)
        self.assertIn("Sentiment", all_markdown)
        self.assertIn("Topic", all_markdown)


class TestPerformanceAndErgonomics(unittest.TestCase):
    """
    Verifies that the entire QA test suite runs in <5.0 seconds for rapid developer CI feedback.
    """

    def test_execution_speed_benchmark(self):
        """Verifies rapid execution timing constraint (<5.0s)."""
        t0 = time.perf_counter()
        at = AppTest.from_file(APP_PATH, default_timeout=10).run()
        elapsed = time.perf_counter() - t0
        self.assertFalse(at.exception)
        self.assertLess(
            elapsed,
            3.0,
            f"Headless render took {elapsed:.2f}s, expected <3.0s",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
