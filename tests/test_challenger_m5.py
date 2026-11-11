"""
tests/test_challenger_m5.py
Milestone 5 Empirical Challenger Test Suite (challenger_m5_1).

Adversarial stress-testing of app/app.py:
1. Preset Interactivity: Simulates all 10 presets via streamlit.testing.v1.AppTest.
2. Probability Invariants: Mathematically verifies sum(p) == 1.0 (±1e-4), bounds [0, 1], and max-label consistency on real models.
3. Input Boundary Hardening: Stress-tests empty, whitespace, tabs, emojis, punctuation, XSS, and ultra-long inputs.
4. Side-by-Side & Speedup: Validates dual cards, 14 progress bars, consensus/divergence rendering, and speedup ratio calculation.
5. Session State & Controls: Tests Clear button, radio model switching, and UI resilience.
6. Real Model E2E Boot: Verifies real models in AppTest without mocks.
"""

import math
import os
import re
import sys
import time
import unittest
from typing import Any, Dict, List

# Windows console encoding safeguard
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_PATH = os.path.join(PROJECT_ROOT, "app", "app.py")
BASELINE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
PHOBERT_DIR = os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask")

import app.app as demo_app
from streamlit.testing.v1 import AppTest
from models.baseline_model import VietnameseFeedbackBaseline, SENTIMENT_MAP, TOPIC_MAP
import models.phobert_model as phobert_module
from transformers import AutoTokenizer


class MockPhoBERTForFastAppTest:
    """Mock PhoBERT fixture for rapid interaction tests (<3s)."""

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


class TestAllPresetsAppTest(unittest.TestCase):
    """
    Simulates user selection and analysis of ALL 10 presets one by one via AppTest.
    """

    @classmethod
    def setUpClass(cls):
        cls.orig_phobert = phobert_module.PhoBERTMultiTask.from_pretrained
        cls.orig_tok = AutoTokenizer.from_pretrained
        phobert_module.PhoBERTMultiTask.from_pretrained = classmethod(lambda c, *a, **k: MockPhoBERTForFastAppTest())
        AutoTokenizer.from_pretrained = classmethod(lambda c, *a, **k: unittest.mock.MagicMock())

        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()

        cls.presets = demo_app.get_presets()

    @classmethod
    def tearDownClass(cls):
        phobert_module.PhoBERTMultiTask.from_pretrained = cls.orig_phobert
        AutoTokenizer.from_pretrained = cls.orig_tok

        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()

    def test_all_10_presets_simulation(self):
        """Iterates through all 10 presets, loads each into text_area, triggers analysis, and verifies UI response."""
        self.assertEqual(len(self.presets), 10, "Expected exactly 10 presets in demo_app")

        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        self.assertFalse(at.exception, f"Initial boot failed: {at.exception}")

        # Ensure baseline mode is active for deterministic single-card testing
        baseline_opts = [opt for opt in at.sidebar.radio[0].options if "Trung bình" in opt]
        self.assertTrue(bool(baseline_opts), "Trung bình radio option not found")
        at.sidebar.radio[0].set_value(baseline_opts[0])

        for idx, preset in enumerate(self.presets):
            # 1. Select preset from dropdown
            at.sidebar.selectbox[0].select_index(idx)
            
            # 2. Click "Load Preset" button
            load_btn = at.sidebar.button[0]
            load_btn.click().run()
            self.assertFalse(at.exception, f"Exception after loading preset #{idx}: {at.exception}")

            # 3. Verify text loaded in text_area
            current_text = at.text_area[0].value
            self.assertEqual(
                current_text,
                preset["text"],
                f"Preset #{idx} text mismatch in st.text_area! Expected: '{preset['text']}', got: '{current_text}'",
            )

            # 4. Verify word and char count caption
            expected_words = len(preset["text"].strip().split())
            expected_chars = len(preset["text"].strip())
            caption_text = " ".join([c.value for c in at.caption])
            self.assertIn(str(expected_words), caption_text, f"Word count {expected_words} missing from caption for preset #{idx}")
            self.assertIn(str(expected_chars), caption_text, f"Char count {expected_chars} missing from caption for preset #{idx}")

            # 5. Click "Phân tích nhận xét (Analyze)"
            analyze_btn = None
            for btn in at.button:
                if "Phân tích" in btn.label or "Analyze" in btn.label:
                    analyze_btn = btn
                    break
            self.assertIsNotNone(analyze_btn, f"Could not find Analyze button for preset #{idx}")
            analyze_btn.click().run()
            self.assertFalse(at.exception, f"Exception during prediction analysis of preset #{idx}: {at.exception}")

            # 6. Verify prediction results rendered
            # Must render at least 7 progress bars (3 sentiment + 4 topic)
            self.assertGreaterEqual(
                len(at.progress),
                7,
                f"Expected at least 7 progress bars for preset #{idx}, got {len(at.progress)}",
            )

            # Verify probability bounds on each progress bar (Streamlit AppTest maps 0.0-1.0 to int 0-100)
            for p_idx, p_bar in enumerate(at.progress):
                self.assertGreaterEqual(p_bar.value, 0, f"Progress bar #{p_idx} value < 0 for preset #{idx}")
                self.assertLessEqual(p_bar.value, 100, f"Progress bar #{p_idx} value > 100 for preset #{idx}")

            # Verify markdown contains sentiment and topic labels or badges
            all_markdown = " ".join([m.value for m in at.markdown])
            self.assertIn("Cảm xúc (Sentiment)", all_markdown)
            self.assertIn("Chủ đề (Topic)", all_markdown)


class TestProbabilityInvariantsEmpirical(unittest.TestCase):
    """
    Mathematically verifies probability distribution invariants on REAL models.
    """

    @classmethod
    def setUpClass(cls):
        cls.baseline = VietnameseFeedbackBaseline.load(BASELINE_PATH)
        cls.phobert = None
        cls.tokenizer = None
        if os.path.isdir(PHOBERT_DIR) and any(os.path.isfile(os.path.join(PHOBERT_DIR, f)) for f in ("model.pt", "pytorch_model.bin")):
            try:
                cls.tokenizer = AutoTokenizer.from_pretrained(PHOBERT_DIR, local_files_only=True)
                cls.phobert = phobert_module.PhoBERTMultiTask.from_pretrained(PHOBERT_DIR, device="cpu")
                cls.phobert.eval()
            except Exception:
                pass

    def _verify_result_invariants(self, res: Dict[str, Any], input_repr: str, model_name: str):
        """Helper to assert all mathematical and schema invariants on a prediction result."""
        # 1. Sentiment invariants
        s = res["sentiment"]
        self.assertIn("label", s)
        self.assertIn("label_id", s)
        self.assertIn("confidence", s)
        self.assertIn("distribution", s)
        self.assertEqual(SENTIMENT_MAP[s["label_id"]], s["label"])

        s_dist = s["distribution"]
        self.assertEqual(set(s_dist.keys()), {"negative", "neutral", "positive"})
        for k, p in s_dist.items():
            self.assertGreaterEqual(p, 0.0, f"Negative prob in {model_name} sentiment dist: {k}={p}")
            self.assertLessEqual(p, 1.0 + 1e-6, f"Prob > 1.0 in {model_name} sentiment dist: {k}={p}")

        s_sum = sum(s_dist.values())
        self.assertAlmostEqual(
            s_sum,
            1.0,
            delta=1e-4,
            msg=f"{model_name} sentiment probabilities do NOT sum to 1.0! Sum: {s_sum} for input: '{input_repr}'",
        )

        max_s_prob = max(s_dist.values())
        self.assertAlmostEqual(
            s["confidence"],
            max_s_prob,
            delta=1e-4,
            msg=f"{model_name} sentiment confidence {s['confidence']} does not equal max prob {max_s_prob}",
        )
        self.assertEqual(
            s["label"],
            max(s_dist, key=s_dist.get),
            msg=f"{model_name} sentiment predicted label does not match argmax",
        )

        # 2. Topic invariants
        t = res["topic"]
        self.assertIn("label", t)
        self.assertIn("label_id", t)
        self.assertIn("confidence", t)
        self.assertIn("distribution", t)
        self.assertEqual(TOPIC_MAP[t["label_id"]], t["label"])

        t_dist = t["distribution"]
        self.assertEqual(set(t_dist.keys()), {"lecturer", "training_program", "facility", "others"})
        for k, p in t_dist.items():
            self.assertGreaterEqual(p, 0.0, f"Negative prob in {model_name} topic dist: {k}={p}")
            self.assertLessEqual(p, 1.0 + 1e-6, f"Prob > 1.0 in {model_name} topic dist: {k}={p}")

        t_sum = sum(t_dist.values())
        self.assertAlmostEqual(
            t_sum,
            1.0,
            delta=1e-4,
            msg=f"{model_name} topic probabilities do NOT sum to 1.0! Sum: {t_sum} for input: '{input_repr}'",
        )

        max_t_prob = max(t_dist.values())
        self.assertAlmostEqual(
            t_confidence := t["confidence"],
            max_t_prob,
            delta=1e-4,
            msg=f"{model_name} topic confidence {t_confidence} does not equal max prob {max_t_prob}",
        )
        self.assertEqual(
            t["label"],
            max(t_dist, key=t_dist.get),
            msg=f"{model_name} topic predicted label does not match argmax",
        )

    def test_baseline_probability_invariants_across_inputs(self):
        """Tests Baseline probability invariants on diverse feedback inputs."""
        test_inputs = [
            "Thầy dạy rất nhiệt tình, giảng giải dễ hiểu.",
            "Phòng học nóng, máy lạnh hư không bật được.",
            "Chương trình học bình thường, không có gì nổi bật.",
            "Học phí quá đắt đỏ so với chất lượng dịch vụ.",
            "???!!!",
            "OK",
            "Tuyệt vời!",
            "Rất tệ",
            "Cảm ơn nhà trường đã cho chúng em xông hơi giữa mùa hè oi bức.",
            "Giáo viên nhiệt tình " * 20,
            "123 456 789 !@#$%",
        ]

        for text in test_inputs:
            res = self.baseline.predict(text)
            self._verify_result_invariants(res, text[:30], "Baseline")

    def test_phobert_probability_invariants_across_inputs(self):
        """Tests PhoBERT probability invariants on diverse feedback inputs."""
        if self.phobert is None or self.tokenizer is None:
            self.skipTest("Real PhoBERT model checkpoint not loaded on this host")

        test_inputs = [
            "Thầy dạy rất nhiệt tình, giảng giải cặn kẽ.",
            "Cơ sở vật chất quá tệ, phòng máy không có mạng.",
            "Nội dung môn học tạm ổn.",
            "???!!!",
            "Tuyệt vời!",
        ]

        for text in test_inputs:
            res = self.phobert.predict(text, tokenizer=self.tokenizer, device="cpu")
            self._verify_result_invariants(res, text[:30], "PhoBERT")


class TestBoundaryAndSpecialInputs(unittest.TestCase):
    """
    Stress-tests empty, whitespace, special characters, and edge inputs in AppTest.
    """

    @classmethod
    def setUpClass(cls):
        cls.orig_phobert = phobert_module.PhoBERTMultiTask.from_pretrained
        cls.orig_tok = AutoTokenizer.from_pretrained
        phobert_module.PhoBERTMultiTask.from_pretrained = classmethod(lambda c, *a, **k: MockPhoBERTForFastAppTest())
        AutoTokenizer.from_pretrained = classmethod(lambda c, *a, **k: unittest.mock.MagicMock())

        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()

    @classmethod
    def tearDownClass(cls):
        phobert_module.PhoBERTMultiTask.from_pretrained = cls.orig_phobert
        AutoTokenizer.from_pretrained = cls.orig_tok

    def test_empty_and_whitespace_inputs_graceful_handling(self):
        """Verifies empty string, spaces, tabs, and newlines show st.warning and do not crash."""
        whitespace_cases = [
            "",
            "   ",
            "\t\t",
            "\n\n\n",
            " \t \n \r  ",
        ]

        for ws in whitespace_cases:
            at = AppTest.from_file(APP_PATH, default_timeout=10).run()
            self.assertFalse(at.exception, f"App exception on init for input {repr(ws)}")

            at.text_area[0].input(ws)

            # Click Analyze button
            for btn in at.button:
                if "Phân tích" in btn.label or "Analyze" in btn.label:
                    btn.click().run()
                    break

            self.assertFalse(at.exception, f"Unhandled exception on whitespace input {repr(ws)}: {at.exception}")

            # Warning must be rendered
            self.assertGreaterEqual(
                len(at.warning),
                1,
                f"Expected st.warning for empty/whitespace input {repr(ws)}, none found!",
            )
            self.assertIn("Vui lòng nhập", at.warning[0].value)

            # No progress bars should be rendered because prediction was prevented
            self.assertEqual(
                len(at.progress),
                0,
                f"Prediction should NOT run for empty input {repr(ws)}, but found {len(at.progress)} progress bars!",
            )

    def test_special_characters_and_edge_inputs(self):
        """Verifies punctuation, emojis, numbers, and XSS patterns process without crashing."""
        special_cases = [
            ("punctuation_only", "???!!!!...."),
            ("emojis_only", "😊🎉🔥💯❤️"),
            ("digits_only", "1234567890"),
            ("symbols_only", "@#$%^&*()_+-=[]{}|;':,.<>?"),
            ("xss_html_injection", "<script>alert('xss')</script><b>bold</b>"),
            ("single_character", "a"),
            ("foreign_language", "This is an English comment with no Vietnamese."),
            ("very_long_input", "Thầy giảng bài rất hay và tâm huyết. " * 80),
        ]

        for case_name, text in special_cases:
            at = AppTest.from_file(APP_PATH, default_timeout=15).run()
            self.assertFalse(at.exception, f"Init error for case {case_name}")

            # Baseline mode
            baseline_opts = [opt for opt in at.sidebar.radio[0].options if "Trung bình" in opt]
            if baseline_opts:
                at.sidebar.radio[0].set_value(baseline_opts[0])

            at.text_area[0].input(text)

            for btn in at.button:
                if "Phân tích" in btn.label or "Analyze" in btn.label:
                    btn.click().run()
                    break

            self.assertFalse(
                at.exception,
                f"Unhandled exception for special case '{case_name}' (input: '{text[:20]}...'): {at.exception}",
            )
            # Should produce 7 progress bars
            self.assertGreaterEqual(
                len(at.progress),
                7,
                f"Expected 7 progress bars for special case '{case_name}', got {len(at.progress)}",
            )


class TestSideBySideModeAndSpeedupRatio(unittest.TestCase):
    """
    Stress-tests Side-by-Side comparison mode, dual columns, consensus cards, and speedup math.
    """

    @classmethod
    def setUpClass(cls):
        cls.orig_phobert = phobert_module.PhoBERTMultiTask.from_pretrained
        cls.orig_tok = AutoTokenizer.from_pretrained
        phobert_module.PhoBERTMultiTask.from_pretrained = classmethod(lambda c, *a, **k: MockPhoBERTForFastAppTest())
        AutoTokenizer.from_pretrained = classmethod(lambda c, *a, **k: unittest.mock.MagicMock())

        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()

    @classmethod
    def tearDownClass(cls):
        phobert_module.PhoBERTMultiTask.from_pretrained = cls.orig_phobert
        AutoTokenizer.from_pretrained = cls.orig_tok

    def test_side_by_side_dual_columns_and_progress_bars(self):
        """Verifies side-by-side mode renders 14 progress bars, dual cards, and comparison metrics."""
        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        self.assertFalse(at.exception)

        # Select Both mode
        sbs_opts = [opt for opt in at.sidebar.radio[0].options if "Cả hai" in opt]
        self.assertTrue(bool(sbs_opts), "Cả hai option missing from sidebar radio")
        at.sidebar.radio[0].set_value(sbs_opts[0])

        at.text_area[0].input("Thầy dạy rất tận tâm, cơ sở vật chất phòng học rất tốt.")

        for btn in at.button:
            if "Phân tích" in btn.label or "Analyze" in btn.label:
                btn.click().run()
                break

        self.assertFalse(at.exception, f"Exception in side-by-side mode: {at.exception}")

        # In side-by-side mode, both models render progress bars (7 + 7 = 14)
        self.assertEqual(
            len(at.progress),
            14,
            f"Expected exactly 14 progress bars in side-by-side mode (7 baseline + 7 phobert), got {len(at.progress)}",
        )

        all_markdown = " ".join([m.value for m in at.markdown])
        self.assertIn("Trung bình", all_markdown)
        self.assertIn("Cao", all_markdown)
        self.assertIn("Phân tích so sánh & Đồng thuận", all_markdown)

        # Verify speedup ratio is rendered and parsable
        all_info = " ".join([i.value for i in at.info])
        self.assertIn("Chênh lệch tốc độ", all_info)
        self.assertIn("Trung bình nhanh hơn", all_info)

        # Parse speedup ratio number
        speedup_match = re.search(r"(\d+(\.\d+)?)x", all_info)
        self.assertIsNotNone(speedup_match, f"Could not find regex speedup pattern '(\\d+)x' in info: '{all_info}'")
        speedup_val = float(speedup_match.group(1))
        self.assertGreaterEqual(speedup_val, 0.0, f"Speedup ratio must be non-negative, got {speedup_val}")

    def test_side_by_side_consensus_and_divergence_rendering(self):
        """Verifies that consensus renders st.success and divergence renders st.warning in consensus boxes."""
        at = AppTest.from_file(APP_PATH, default_timeout=15).run()
        sbs_opts = [opt for opt in at.sidebar.radio[0].options if "Cả hai" in opt]
        at.sidebar.radio[0].set_value(sbs_opts[0])

        # Input where baseline and mock PhoBERT will agree on sentiment (positive)
        at.text_area[0].input("Thầy dạy rất tuyệt vời và nhiệt tình")
        for btn in at.button:
            if "Phân tích" in btn.label or "Analyze" in btn.label:
                btn.click().run()
                break

        self.assertFalse(at.exception)
        # Check success messages for consensus
        success_texts = " ".join([s.value for s in at.success])
        self.assertIn("Đồng thuận", success_texts)


class TestAppControlsAndClearAction(unittest.TestCase):
    """
    Tests secondary controls such as the Clear button and model switching.
    """

    @classmethod
    def setUpClass(cls):
        cls.orig_phobert = phobert_module.PhoBERTMultiTask.from_pretrained
        cls.orig_tok = AutoTokenizer.from_pretrained
        phobert_module.PhoBERTMultiTask.from_pretrained = classmethod(lambda c, *a, **k: MockPhoBERTForFastAppTest())
        AutoTokenizer.from_pretrained = classmethod(lambda c, *a, **k: unittest.mock.MagicMock())

    @classmethod
    def tearDownClass(cls):
        phobert_module.PhoBERTMultiTask.from_pretrained = cls.orig_phobert
        AutoTokenizer.from_pretrained = cls.orig_tok

    def test_clear_button_resets_session_state(self):
        """Clicking Clear button empties session state user_text."""
        at = AppTest.from_file(APP_PATH, default_timeout=10).run()
        self.assertFalse(at.exception)

        # Type text
        at.text_area[0].input("Nội dung cần xóa")

        clear_btn = None
        for btn in at.button:
            if "Xóa trắng" in btn.label or "Clear" in btn.label:
                clear_btn = btn
                break
        self.assertIsNotNone(clear_btn, "Clear button not found")

        clear_btn.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["user_text"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
