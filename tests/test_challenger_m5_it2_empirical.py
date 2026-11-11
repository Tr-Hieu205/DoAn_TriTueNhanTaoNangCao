"""
tests/test_challenger_m5_it2_empirical.py
Empirical stress-testing and fault tolerance harness by challenger_m5_it2_2.

Focus areas:
1. Challenge 1 verification: Missing PhoBERT checkpoint renders st.warning with train command.
2. Challenge 3 verification: CUDA OOM during forward pass in both Single Mode and Side-by-Side mode.
3. Edge case inputs: empty feedback, whitespace only, unicode symbols/emojis, 5000+ chars text.
4. App crash-free guarantee under adverse conditions (no unhandled exceptions).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_PATH = os.path.join(PROJECT_ROOT, "app", "app.py")

import streamlit as st
from streamlit.testing.v1 import AppTest
import app.app as demo_app
import models.phobert_model as phobert_module
import torch


class TestChallengerM5It2Empirical(unittest.TestCase):
    """Deep adversarial and edge case verification suite."""

    def setUp(self):
        st.cache_resource.clear()
        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()

    def tearDown(self):
        st.cache_resource.clear()
        if hasattr(demo_app.load_phobert_pipeline, "clear"):
            demo_app.load_phobert_pipeline.clear()
        if hasattr(demo_app.load_baseline_model, "clear"):
            demo_app.load_baseline_model.clear()

    def test_challenge_1_phobert_missing_banner_and_clean_degrade(self):
        """Verify st.warning and 'python training/train_phobert.py' on missing PhoBERT weights."""
        with patch.object(
            phobert_module.PhoBERTMultiTask,
            "from_pretrained",
            side_effect=FileNotFoundError("No checkpoint found"),
        ):
            at = AppTest.from_file(APP_PATH, default_timeout=10).run()
            self.assertFalse(at.exception, f"App crashed when PhoBERT missing: {at.exception}")

            # Check that warning is rendered
            warning_texts = [w.value for w in at.warning]
            self.assertTrue(len(warning_texts) > 0, "Expected at least one st.warning banner")

            banner = warning_texts[0]
            self.assertIn("Chưa tìm thấy checkpoint mô hình Cao (PhoBERT)", banner)
            self.assertIn("python training/train_phobert.py", banner)
            self.assertIn("Trung bình", banner)

            # Check sidebar shows status correctly
            sidebar_text = " ".join([m.value for m in at.sidebar.markdown])
            self.assertIn("Cao: `❌ Thiếu weights`", sidebar_text)
            self.assertIn("Trung bình: `✅ Sẵn sàng`", sidebar_text)

    def test_challenge_3_cuda_oom_single_mode_failover(self):
        """Verify CUDA OOM triggers torch.cuda.empty_cache, model.to('cpu'), st.warning, and CPU re-inference in single PhoBERT mode."""
        empty_cache_called = False
        to_cpu_called = False
        devices_called = []

        def mock_empty_cache():
            nonlocal empty_cache_called
            empty_cache_called = True

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        def mock_to(device):
            nonlocal to_cpu_called
            if device == "cpu":
                to_cpu_called = True
            return mock_model

        mock_model.to = mock_to

        def mock_predict(text, tokenizer=None, device="cuda", **kwargs):
            devices_called.append(str(device))
            if str(device) == "cuda":
                raise torch.cuda.OutOfMemoryError("Mock CUDA OOM in test")
            return {
                "sentiment": {
                    "label": "positive",
                    "label_id": 2,
                    "confidence": 0.98,
                    "distribution": {"negative": 0.01, "neutral": 0.01, "positive": 0.98},
                },
                "topic": {
                    "label": "lecturer",
                    "label_id": 0,
                    "confidence": 0.95,
                    "distribution": {
                        "lecturer": 0.95,
                        "training_program": 0.02,
                        "facility": 0.02,
                        "others": 0.01,
                    },
                },
            }

        mock_model.predict = mock_predict

        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "empty_cache", side_effect=mock_empty_cache):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()):
                    with patch.object(
                        phobert_module.PhoBERTMultiTask, "from_pretrained", return_value=mock_model
                    ):
                        at = AppTest.from_file(APP_PATH, default_timeout=10).run()
                        self.assertFalse(at.exception)

                        # Select PhoBERT mode
                        phobert_opts = [
                            opt for opt in at.sidebar.radio[0].options if "Cao" in opt
                        ]
                        at.sidebar.radio[0].set_value(phobert_opts[0])

                        # Enter text and analyze
                        at.text_area[0].input("Thầy giảng dạy rất dễ hiểu.")
                        analyze_btn = next((b for b in at.button if "Phân tích" in b.label), None)
                        self.assertIsNotNone(analyze_btn)
                        analyze_btn.click().run()

                        self.assertFalse(at.exception, f"App crashed on CUDA OOM: {at.exception}")
                        self.assertTrue(empty_cache_called, "empty_cache was NOT called!")
                        self.assertTrue(to_cpu_called, "model.to('cpu') was NOT called!")
                        self.assertEqual(devices_called, ["cuda", "cpu"])

                        # Warning banner must be present
                        warning_texts = [w.value for w in at.warning]
                        self.assertTrue(
                            any("CUDA Out of Memory" in w for w in warning_texts),
                            f"Expected CUDA OOM warning banner, got: {warning_texts}",
                        )

    def test_challenge_3_cuda_oom_side_by_side_mode_failover(self):
        """Verify CUDA OOM triggers cache clear, CPU failover, and warning during Side-by-Side comparison mode."""
        empty_cache_called = False
        to_cpu_called = False
        phobert_devices = []

        def mock_empty_cache():
            nonlocal empty_cache_called
            empty_cache_called = True

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model

        def mock_to(device):
            nonlocal to_cpu_called
            if device == "cpu":
                to_cpu_called = True
            return mock_model

        mock_model.to = mock_to

        def mock_predict(text, tokenizer=None, device="cuda", **kwargs):
            phobert_devices.append(str(device))
            if str(device) == "cuda":
                raise torch.cuda.OutOfMemoryError("Mock CUDA OOM in side-by-side test")
            return {
                "sentiment": {
                    "label": "positive",
                    "label_id": 2,
                    "confidence": 0.95,
                    "distribution": {"negative": 0.02, "neutral": 0.03, "positive": 0.95},
                },
                "topic": {
                    "label": "lecturer",
                    "label_id": 0,
                    "confidence": 0.90,
                    "distribution": {
                        "lecturer": 0.90,
                        "training_program": 0.05,
                        "facility": 0.03,
                        "others": 0.02,
                    },
                },
            }

        mock_model.predict = mock_predict

        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.cuda, "empty_cache", side_effect=mock_empty_cache):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()):
                    with patch.object(
                        phobert_module.PhoBERTMultiTask, "from_pretrained", return_value=mock_model
                    ):
                        at = AppTest.from_file(APP_PATH, default_timeout=10).run()
                        self.assertFalse(at.exception)

                        # Select Side-by-Side mode
                        sbs_opts = [
                            opt for opt in at.sidebar.radio[0].options if "Cả hai" in opt
                        ]
                        self.assertTrue(len(sbs_opts) > 0)
                        at.sidebar.radio[0].set_value(sbs_opts[0])

                        # Enter text and analyze
                        at.text_area[0].input("Thầy dạy rất tận tình, phòng học mát mẻ.")
                        analyze_btn = next((b for b in at.button if "Phân tích" in b.label), None)
                        self.assertIsNotNone(analyze_btn)
                        analyze_btn.click().run()

                        self.assertFalse(at.exception, f"App crashed in Side-by-Side on CUDA OOM: {at.exception}")
                        self.assertTrue(empty_cache_called, "empty_cache was NOT called in SBS mode!")
                        self.assertTrue(to_cpu_called, "model.to('cpu') was NOT called in SBS mode!")
                        self.assertEqual(phobert_devices, ["cuda", "cpu"])

                        # Warning banner must be present
                        warning_texts = [w.value for w in at.warning]
                        self.assertTrue(
                            any("CUDA Out of Memory" in w for w in warning_texts),
                            f"Expected CUDA OOM warning banner in SBS, got: {warning_texts}",
                        )

                        # Must still render both comparison cards (14 progress bars)
                        self.assertEqual(
                            len(at.progress),
                            14,
                            f"Expected 14 progress bars in SBS failover, got {len(at.progress)}",
                        )

    def test_edge_case_empty_and_whitespace_only(self):
        """Stress-test empty feedback and pure whitespace combinations across modes."""
        whitespace_payloads = [
            "",
            "   ",
            "\t\t\t",
            "\n\n\n\n",
            "   \t  \n  \r\n   ",
            "         \u00a0        ",   # Non-breaking space
        ]

        for ws in whitespace_payloads:
            at = AppTest.from_file(APP_PATH, default_timeout=10).run()
            self.assertFalse(at.exception)

            at.text_area[0].input(ws)
            analyze_btn = next((b for b in at.button if "Phân tích" in b.label), None)
            self.assertIsNotNone(analyze_btn)
            analyze_btn.click().run()

            # Must not crash
            self.assertFalse(at.exception, f"Crashed on whitespace payload: {repr(ws)}")
            # Must show validation warning
            warning_texts = [w.value for w in at.warning]
            self.assertTrue(
                any("Vui lòng nhập nội dung nhận xét" in w for w in warning_texts),
                f"Missing validation warning for: {repr(ws)}",
            )
            # Must NOT execute prediction progress bars
            self.assertEqual(len(at.progress), 0)

    def test_edge_case_unicode_symbols_and_emojis(self):
        """Stress-test unicode symbols, emojis, math notation, zero-width chars, and stacked diacritics."""
        unicode_payloads = [
            "👍✨🔥🚀💯💔😢🎉🎊 𠜎 𠜱 𠝹 𠱓 𠱸 𠲖 𠳏",  # Emojis & CJK extension B
            "\u200b\u200c\u200d\ufeff",  # Zero-width format characters
            "∑ ∏ ∫ ∬ ∭ ∮ ∇ ∆ ∂ ∀ ∃ ∄ ∈ ∉ ∋ ∌ ⊆ ⊇ ⊂ ⊃ ∪ ∩ ∧ ∨ ¬ ⇒ ⇔",  # Math symbols
            "ệ́ ề́ ể́ ễ́ ợ́ ớ́ ứ́ ừ́ ử́ ữ́",  # Combining diacritics stacking
            "«»‹›“”‘’—–…§¶†‡©®™°±×÷",  # Typographic glyphs
            "👩‍🏫 👨‍🎓 💻 📚 🏫 🎓 📝",  # Multi-codepoint ZWJ emoji sequences
        ]

        for payload in unicode_payloads:
            at = AppTest.from_file(APP_PATH, default_timeout=10).run()
            self.assertFalse(at.exception)

            at.text_area[0].input(payload)
            analyze_btn = next((b for b in at.button if "Phân tích" in b.label), None)
            self.assertIsNotNone(analyze_btn)
            analyze_btn.click().run()

            # Must not crash
            self.assertFalse(at.exception, f"Crashed on Unicode payload: {payload}")
            # In default side-by-side mode, renders 14 progress bars
            self.assertGreaterEqual(len(at.progress), 7)

    def test_edge_case_extremely_long_text_5000_plus_chars(self):
        """Stress-test extremely long text inputs (5,000 to 50,000 characters)."""
        long_payloads = [
            ("5k_chars", "Giảng viên hướng dẫn rất tận tâm và giải đáp cặn kẽ mọi thắc mắc. " * 80),  # ~5,360 chars
            ("10k_chars", "Cơ sở vật chất trường học hiện đại, máy móc tân tiến và phòng học sạch đẹp. " * 140),  # ~10,360 chars
            ("30k_chars", "Chương trình đào tạo đáp ứng tốt chuẩn đầu ra doanh nghiệp. " * 500),  # ~30,000 chars
            ("50k_chars", "Phản hồi ý kiến sinh viên kịp thời. " * 1500),  # ~52,500 chars
        ]

        for name, text in long_payloads:
            char_count = len(text)
            self.assertGreaterEqual(char_count, 5000, f"Payload {name} must be >= 5000 chars")

            at = AppTest.from_file(APP_PATH, default_timeout=15).run()
            self.assertFalse(at.exception)

            at.text_area[0].input(text)
            analyze_btn = next((b for b in at.button if "Phân tích" in b.label), None)
            self.assertIsNotNone(analyze_btn)
            analyze_btn.click().run()

            self.assertFalse(
                at.exception,
                f"App crashed on long text payload '{name}' ({char_count} chars): {at.exception}",
            )
            # In SBS mode, results rendered successfully
            self.assertGreaterEqual(len(at.progress), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
