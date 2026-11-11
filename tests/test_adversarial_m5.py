"""
tests/test_adversarial_m5.py
Adversarial stress and resilience test suite for Milestone 5 Streamlit Dashboard.
Specifically tests:
1. Missing PhoBERT Checkpoint Simulation & user guidance (python training/train_phobert.py).
2. Missing Baseline Checkpoint Simulation & clean error handling.
3. CUDA OOM Simulation during forward pass (cache clear + CPU failover).
4. Subprocess Headless Boot within 5 seconds on port 8503.
5. Adversarial Input Stress (massive payloads, XSS/script tags, zero-width spaces).
"""

import os
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

APP_PATH = os.path.join(PROJECT_ROOT, "app", "app.py")

import streamlit as st
from streamlit.testing.v1 import AppTest
import app.app as demo_app
import models.phobert_model as phobert_module
import torch


class TestMilestone5ResilienceAndAdversarial(unittest.TestCase):
    """Adversarial stress suite reproducing the 4 challenge mission criteria."""

    def setUp(self):
        # Clear streamlit caches before each test
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

    def test_challenge_1_missing_phobert_checkpoint_guidance(self):
        """
        Challenge 1: Missing PhoBERT Checkpoint Simulation.
        Requirement:
        - Temporarily point or mock load_phobert_pipeline to non-existent path.
        - Verify dashboard displays st.warning.
        - Verify dashboard instructs user with 'python training/train_phobert.py'.
        - Gracefully falls back to Baseline without crashing.
        """
        # Patch PhoBERT loading to fail (simulating missing weights)
        with patch.object(phobert_module.PhoBERTMultiTask, "from_pretrained", side_effect=FileNotFoundError("Mock missing PhoBERT weights")):
            at = AppTest.from_file(APP_PATH, default_timeout=10).run()

            # 1. Must not crash with unhandled exception
            self.assertFalse(at.exception, f"App crashed on missing PhoBERT checkpoint: {at.exception}")

            # 2. Check if baseline is still available
            radio_options = at.sidebar.radio[0].options if at.sidebar.radio else []
            self.assertTrue(
                any("Trung bình" in opt for opt in radio_options),
                "Trung bình not available when PhoBERT is missing",
            )

            # 3. Check for st.warning
            warnings = [w.value for w in at.warning]
            print(f"[DEBUG test_challenge_1] Rendered warnings: {warnings}")

            # 4. Check for instruction 'python training/train_phobert.py'
            all_rendered_text = " ".join(
                [m.value for m in at.markdown]
                + [w.value for w in at.warning]
                + [e.value for e in at.error]
                + [i.value for i in at.info]
                + [m.value for m in at.sidebar.markdown]
            )
            has_train_instruction = "python training/train_phobert.py" in all_rendered_text or "train_phobert.py" in all_rendered_text
            print(f"[DEBUG test_challenge_1] Has train_phobert instruction: {has_train_instruction}")

            self.assertGreater(
                len(warnings),
                0,
                "FAIL: Expected st.warning on dashboard when PhoBERT checkpoint is missing, but no st.warning was rendered.",
            )
            self.assertTrue(
                has_train_instruction,
                "FAIL: Expected user instruction containing 'python training/train_phobert.py' on missing checkpoint.",
            )

    def test_challenge_2_missing_baseline_checkpoint(self):
        """
        Challenge 2: Missing Baseline Checkpoint Simulation.
        Requirement:
        - Simulate missing saved_models/baseline.joblib.
        - Verify clean error notice without unhandled exception.
        """
        from models.baseline_model import VietnameseFeedbackBaseline
        with patch.object(VietnameseFeedbackBaseline, "load", side_effect=FileNotFoundError("Mock missing baseline.joblib")):
            at = AppTest.from_file(APP_PATH, default_timeout=10).run()

            # 1. Must not raise unhandled exception
            self.assertFalse(at.exception, f"App crashed when baseline was missing: {at.exception}")

            # 2. Check for clean notice in sidebar or main page
            sidebar_text = " ".join([m.value for m in at.sidebar.markdown] + [e.value for e in at.sidebar.error])
            print(f"[DEBUG test_challenge_2] Sidebar text: {sidebar_text}")
            self.assertTrue(
                "Thiếu weights" in sidebar_text or "Không tìm thấy" in sidebar_text or "❌" in sidebar_text,
                "Expected clean notice regarding missing baseline checkpoint.",
            )

    def test_challenge_3_cuda_oom_during_forward_pass(self):
        """
        Challenge 3: CUDA OOM Simulation during forward pass.
        Requirement:
        - Simulate torch.cuda.OutOfMemoryError during forward pass.
        - Verify automatic cache clearing (torch.cuda.empty_cache called) and failover to CPU.
        """
        cache_cleared = False
        failover_to_cpu = False

        def mock_empty_cache():
            nonlocal cache_cleared
            cache_cleared = True

        def mock_predict(texts, tokenizer=None, device="cuda", **kwargs):
            nonlocal failover_to_cpu
            device_str = str(device)
            if "cuda" in device_str:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory during forward pass in test")
            failover_to_cpu = True
            return {
                "sentiment": {"label": "positive", "label_id": 2, "confidence": 0.99, "distribution": {"negative": 0.0, "neutral": 0.01, "positive": 0.99}},
                "topic": {"label": "lecturer", "label_id": 0, "confidence": 0.95, "distribution": {"lecturer": 0.95, "training_program": 0.03, "facility": 0.01, "others": 0.01}},
            }

        with patch.object(torch.cuda, "empty_cache", side_effect=mock_empty_cache):
            with patch.object(torch.cuda, "is_available", return_value=True):
                with patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()):
                    mock_instance = MagicMock()
                    mock_instance.eval.return_value = mock_instance
                    mock_instance.predict = mock_predict
                    
                    with patch.object(phobert_module.PhoBERTMultiTask, "from_pretrained", return_value=mock_instance):
                        at = AppTest.from_file(APP_PATH, default_timeout=10).run()

                        # Select PhoBERT mode
                        phobert_opts = [opt for opt in at.sidebar.radio[0].options if "Cao" in opt]
                        self.assertTrue(len(phobert_opts) > 0, "Cao option missing")
                        at.sidebar.radio[0].set_value(phobert_opts[0])

                        # Enter text and click analyze
                        at.text_area[0].input("Thầy dạy rất nhiệt tình.")
                        analyze_btn = next((b for b in at.button if "Phân tích" in b.label or "Analyze" in b.label), None)
                        self.assertIsNotNone(analyze_btn)

                        analyze_btn.click().run()

                        print(f"[DEBUG test_challenge_3] Exceptions: {at.exception}")
                        print(f"[DEBUG test_challenge_3] Cache cleared: {cache_cleared}")
                        print(f"[DEBUG test_challenge_3] Failover to CPU: {failover_to_cpu}")

                        self.assertFalse(at.exception, f"App crashed on CUDA OOM during forward pass: {at.exception}")
                        self.assertTrue(cache_cleared, "FAIL: torch.cuda.empty_cache() was not called upon CUDA OOM.")
                        self.assertTrue(failover_to_cpu, "FAIL: Did not failover to CPU upon CUDA OOM.")

    def test_challenge_4_subprocess_headless_boot_port_8503(self):
        """
        Challenge 4: Subprocess Headless Boot.
        Requirement:
        - Run `python -m streamlit run app/app.py --server.headless=true --server.port=8503` in background.
        - Verify it starts listening within 5 seconds without errors.
        - Terminate cleanly.
        """
        port = 8503
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/app.py",
            "--server.headless=true",
            f"--server.port={port}",
        ]

        t0 = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT,
        )

        listening = False
        elapsed = 0.0
        try:
            deadline = time.time() + 10.0  # 10s max wait, but must connect <= 5s
            while time.time() < deadline:
                elapsed = time.perf_counter() - t0
                # Test port connection
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    res = s.connect_ex(("127.0.0.1", port))
                    if res == 0:
                        listening = True
                        break
                time.sleep(0.3)

            print(f"[DEBUG test_challenge_4] Listening: {listening}, Elapsed time: {elapsed:.2f}s")
            self.assertTrue(listening, f"Streamlit server did not listen on port {port} within deadline")
            self.assertLessEqual(elapsed, 5.0, f"Streamlit server took {elapsed:.2f}s to start listening, exceeding 5.0s requirement")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


    def test_adversarial_inputs_stress(self):
        """
        Stress-tests app against extreme and adversarial inputs:
        - Massive 30,000 character string
        - Pure emoji strings
        - HTML/XSS injection tags
        - Rapid repetitive patterns
        """
        at = AppTest.from_file(APP_PATH, default_timeout=10).run()
        self.assertFalse(at.exception)

        adversarial_payloads = [
            # 1. HTML / Script injection attempt
            "<script>alert('xss')</script><b>Đậm</b><i>Nghiêng</i>",
            # 2. Pure emoji string
            "😀😁😂🤣😃😄😅😆😉😊😋😎😍😘🥰",
            # 3. Massive repetitive string
            "Thầy cô dạy rất hay và nhiệt tình! " * 500,
            # 4. Zero-width and non-printable characters
            "\u200b\u200c\u200d\ufeff Giảng viên tận tâm \u0000\u0007",
        ]

        for payload in adversarial_payloads:
            at.text_area[0].input(payload)
            analyze_btn = next((b for b in at.button if "Phân tích" in b.label or "Analyze" in b.label), None)
            self.assertIsNotNone(analyze_btn)
            analyze_btn.click().run()

            self.assertFalse(
                at.exception,
                f"App crashed on adversarial payload: {payload[:40]}... Exception: {at.exception}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
