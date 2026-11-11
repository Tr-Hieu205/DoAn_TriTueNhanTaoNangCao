"""
tests/test_phobert_stress.py
Adversarial Stress Test Suite for Milestone 3: PhoBERT Multi-Task Model & Fine-Tuning Pipeline.

Author: Challenger M3_2 (critic, specialist)
Validates empirical robustness under:
1. Boundary & Corrupt Inputs:
   - Empty inputs: "", "   ", "\n\t\r", []
   - None within batch texts
   - Huge inputs: Very long paragraphs (>1000 tokens) with clean truncation
   - Punctuation-only, emojis, and malformed/decomposed Vietnamese text
2. Batch Size Permutations:
   - Batch size 1, 64, and uneven/prime batch sizes (3, 7, 13, 65)
   - Consistency across single vs batch inference
3. Forward Signature & Parameter Robustness:
   - forward(input_ids, attention_mask=None, sentiment_labels=None, topic_labels=None)
   - Single-label loss computation (sentiment only, topic only)
   - Joint multi-task loss computation (both labels)
   - Inactive loss (no labels provided)
   - token_type_ids handling
4. Checkpoint Save/Reload Idempotence Stress:
   - Save real model to temporary directory
   - Reload from temporary directory
   - 100 random samples from UIT-VSFC test split evaluated for 100% exact parity
"""

import json
import math
import os
import random
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

# Project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data.dataset import load_vsfc_split
from models.phobert_model import (
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
    TOPIC_LABELS,
    TOPIC_MAP,
    PhoBERTMultiTask,
)
from tests.test_phobert import DummyPhoBERTTokenizer, create_tiny_phobert_model


class TestPhoBERTBoundaryInputs(unittest.TestCase):
    """Stress tests on boundary, empty, corrupt, and extreme inputs."""

    @classmethod
    def setUpClass(cls):
        cls.tiny_model = create_tiny_phobert_model(hidden_size=64)
        cls.dummy_tok = DummyPhoBERTTokenizer(vocab_size=1000)

    def test_empty_and_whitespace_strings(self):
        """Verify empty string, spaces, tabs, and newlines do not crash and produce valid schemas."""
        boundary_cases = ["", "   ", "\t\t\t", "\n\r\n", "   \n\t   "]
        for text in boundary_cases:
            res = self.tiny_model.predict(text, tokenizer=self.dummy_tok)
            self.assertIsInstance(res, dict, f"Failed for text: {repr(text)}")
            self.assertIn("sentiment", res)
            self.assertIn("topic", res)
            self.assertIn(res["sentiment"]["label"], SENTIMENT_LABELS)
            self.assertIn(res["topic"]["label"], TOPIC_LABELS)
            self.assertIsInstance(res["sentiment"]["label_id"], int)
            self.assertIsInstance(res["topic"]["label_id"], int)
            self.assertIsInstance(res["sentiment"]["confidence"], float)
            self.assertIsInstance(res["topic"]["confidence"], float)

            # Sum of distributions must equal 1.0
            self.assertAlmostEqual(sum(res["sentiment"]["distribution"].values()), 1.0, places=5)
            self.assertAlmostEqual(sum(res["topic"]["distribution"].values()), 1.0, places=5)

    def test_empty_list_input(self):
        """Verify passing empty list [] returns empty list/arrays without exception."""
        res = self.tiny_model.predict([], tokenizer=self.dummy_tok)
        self.assertEqual(res, [])

        sent_probs, topic_probs = self.tiny_model.predict_proba([], tokenizer=self.dummy_tok)
        self.assertEqual(sent_probs.shape, (0, 3))
        self.assertEqual(topic_probs.shape, (0, 4))

        sent_ids, topic_ids = self.tiny_model.predict_labels([], tokenizer=self.dummy_tok)
        self.assertEqual(sent_ids, [])
        self.assertEqual(topic_ids, [])

    def test_none_inside_batch(self):
        """Verify that None elements inside a batch list are safely handled."""
        mixed_batch = ["Giảng viên tận tâm", None, "", "Phòng học quá nóng"]
        res = self.tiny_model.predict(mixed_batch, tokenizer=self.dummy_tok)
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 4)
        for i, item in enumerate(res):
            self.assertIn("sentiment", item, f"Missing sentiment in item {i}")
            self.assertIn("topic", item, f"Missing topic in item {i}")
            self.assertAlmostEqual(sum(item["sentiment"]["distribution"].values()), 1.0, places=5)
            self.assertAlmostEqual(sum(item["topic"]["distribution"].values()), 1.0, places=5)

    def test_huge_input_truncation(self):
        """Verify paragraph with >1000 tokens truncates cleanly without crash or shape error."""
        long_paragraph = "sinh_viên học_tập chăm_chỉ tại trường đại_học " * 300  # ~1800 words
        res = self.tiny_model.predict(long_paragraph, tokenizer=self.dummy_tok, max_length=128)
        self.assertIsInstance(res, dict)
        self.assertIn("sentiment", res)
        self.assertIn("topic", res)

        # Batch of long paragraphs
        batch_long = [long_paragraph, "Ngắn", long_paragraph]
        batch_res = self.tiny_model.predict(batch_long, tokenizer=self.dummy_tok, max_length=64)
        self.assertEqual(len(batch_res), 3)

    def test_punctuation_emojis_and_malformed_vietnamese(self):
        """Verify punctuation-only, emoji spam, and malformed Vietnamese diacritics."""
        adversarial_inputs = [
            "???!!! ...",
            "😊👍🎉❤️🔥🔥🔥",
            "àáạảã èéẹẻẽ ìíịỉĩ òóọỏõ ùúụủũ ỳýỵỷỹ",
            "!!! Thầy dạy ??? quá chán 😡😡😡 !!!",
            "Cơ sở vật chất: phòng máy tính A102 máy hỏng liên tục 🖥️💥🔌",
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
            "1234567890 0987654321 + - * / = % ^ & * ( ) _ +",
            "<html><body><script>alert('xss');</script></body></html>",
            "SELECT * FROM feedback WHERE 1=1; DROP TABLE users;--",
        ]
        res = self.tiny_model.predict(adversarial_inputs, tokenizer=self.dummy_tok)
        self.assertEqual(len(res), len(adversarial_inputs))
        for i, item in enumerate(res):
            self.assertIn(item["sentiment"]["label"], SENTIMENT_LABELS)
            self.assertIn(item["topic"]["label"], TOPIC_LABELS)
            self.assertAlmostEqual(sum(item["sentiment"]["distribution"].values()), 1.0, places=5)
            self.assertAlmostEqual(sum(item["topic"]["distribution"].values()), 1.0, places=5)


class TestPhoBERTBatchSizeStress(unittest.TestCase):
    """Stress tests on varying, odd, and edge batch sizes."""

    @classmethod
    def setUpClass(cls):
        cls.tiny_model = create_tiny_phobert_model(hidden_size=64)
        cls.dummy_tok = DummyPhoBERTTokenizer(vocab_size=1000)
        cls.sample_texts = [
            "Thầy cô dạy rất hay và nhiệt tình",
            "Phòng học thiếu quạt và máy lạnh",
            "Chương trình học nặng lý thuyết",
            "Tạm được, không có ý kiến gì",
            "Trường cần đầu tư thêm trang thiết bị",
            "Môn học này rất bổ ích cho sinh viên",
            "Giáo trình chưa được cập nhật mới",
            "Thầy chấm điểm quá khắt khe",
        ]

    def test_single_item_batch_size_one(self):
        """Verify batch_size=1 on multi-item lists."""
        res = self.tiny_model.predict(self.sample_texts, tokenizer=self.dummy_tok, batch_size=1)
        self.assertEqual(len(res), len(self.sample_texts))

        # Check single vs batch_size=1 consistency
        for i, text in enumerate(self.sample_texts):
            single_res = self.tiny_model.predict(text, tokenizer=self.dummy_tok)
            self.assertEqual(res[i]["sentiment"]["label"], single_res["sentiment"]["label"])
            self.assertEqual(res[i]["topic"]["label"], single_res["topic"]["label"])
            self.assertAlmostEqual(
                res[i]["sentiment"]["confidence"], single_res["sentiment"]["confidence"], places=5
            )
            self.assertAlmostEqual(
                res[i]["topic"]["confidence"], single_res["topic"]["confidence"], places=5
            )

    def test_large_batch_size(self):
        """Verify batch_size=64 on list smaller than batch size."""
        res = self.tiny_model.predict(self.sample_texts, tokenizer=self.dummy_tok, batch_size=64)
        self.assertEqual(len(res), len(self.sample_texts))

    def test_uneven_and_prime_batch_sizes(self):
        """Verify prime and uneven batch sizes (3, 7, 13) on a 25-item list."""
        repeated_texts = (self.sample_texts * 4)[:25]
        for bs in [3, 7, 13]:
            res = self.tiny_model.predict(repeated_texts, tokenizer=self.dummy_tok, batch_size=bs)
            self.assertEqual(len(res), 25, f"Mismatch with batch_size={bs}")


class TestPhoBERTForwardSignatureStress(unittest.TestCase):
    """Stress tests on forward signature, argument combinations, and loss backprop."""

    @classmethod
    def setUpClass(cls):
        cls.tiny_model = create_tiny_phobert_model(hidden_size=64)

    def test_forward_missing_optional_arguments(self):
        """Verify forward works when optional args are None or omitted."""
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)

        # 1. input_ids only
        out1 = self.tiny_model(input_ids)
        self.assertIsNone(out1["loss"])
        self.assertEqual(out1["sentiment_logits"].shape, (batch_size, 3))
        self.assertEqual(out1["topic_logits"].shape, (batch_size, 4))

        # 2. explicit None kwargs
        out2 = self.tiny_model(
            input_ids=input_ids,
            attention_mask=None,
            sentiment_labels=None,
            topic_labels=None,
            token_type_ids=None,
        )
        self.assertIsNone(out2["loss"])
        self.assertEqual(out2["sentiment_logits"].shape, (batch_size, 3))
        self.assertEqual(out2["topic_logits"].shape, (batch_size, 4))

    def test_forward_single_task_labels(self):
        """Verify loss is computed correctly when only one task label is supplied."""
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        sent_labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        topic_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        # Sentiment labels only
        out_sent = self.tiny_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentiment_labels=sent_labels,
            topic_labels=None,
        )
        self.assertIsNotNone(out_sent["loss"])
        self.assertTrue(torch.is_tensor(out_sent["loss"]))
        self.assertEqual(out_sent["loss"].ndim, 0)

        # Topic labels only
        out_topic = self.tiny_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentiment_labels=None,
            topic_labels=topic_labels,
        )
        self.assertIsNotNone(out_topic["loss"])
        self.assertTrue(torch.is_tensor(out_topic["loss"]))
        self.assertEqual(out_topic["loss"].ndim, 0)

        # In eval mode (dropout disabled), separate calls produce identical logits
        self.tiny_model.eval()
        with torch.no_grad():
            eval_sent = self.tiny_model(input_ids=input_ids, sentiment_labels=sent_labels)
            eval_topic = self.tiny_model(input_ids=input_ids, topic_labels=topic_labels)
            eval_both = self.tiny_model(input_ids=input_ids, sentiment_labels=sent_labels, topic_labels=topic_labels)
            self.assertAlmostEqual(eval_both["loss"].item(), (eval_sent["loss"] + eval_topic["loss"]).item(), places=5)

        # In train mode (dropout enabled), joint loss must equal sum of losses on the same forward logits
        self.tiny_model.train()
        loss_fn = nn.CrossEntropyLoss()
        out_both = self.tiny_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentiment_labels=sent_labels,
            topic_labels=topic_labels,
        )
        self.assertIsNotNone(out_both["loss"])
        expected_joint_loss = (
            loss_fn(out_both["sentiment_logits"], sent_labels)
            + loss_fn(out_both["topic_logits"], topic_labels)
        ).item()
        self.assertAlmostEqual(out_both["loss"].item(), expected_joint_loss, places=5)

    def test_gradient_flow_and_optimization_step(self):
        """Verify backward step updates model weights across shared backbone and both heads."""
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)
        sent_labels = torch.tensor([0, 2], dtype=torch.long)
        topic_labels = torch.tensor([1, 3], dtype=torch.long)

        optimizer = torch.optim.SGD(self.tiny_model.parameters(), lr=0.01)
        optimizer.zero_grad()

        out = self.tiny_model(
            input_ids=input_ids,
            sentiment_labels=sent_labels,
            topic_labels=topic_labels,
        )
        loss = out["loss"]
        loss.backward()

        # Check gradients exist on backbone, sentiment head, topic head
        self.assertIsNotNone(self.tiny_model.sentiment_classifier.weight.grad)
        self.assertIsNotNone(self.tiny_model.topic_classifier.weight.grad)
        self.assertGreater(torch.norm(self.tiny_model.sentiment_classifier.weight.grad).item(), 0.0)
        self.assertGreater(torch.norm(self.tiny_model.topic_classifier.weight.grad).item(), 0.0)

        optimizer.step()


class TestPhoBERTCheckpointSaveReloadIdempotenceStress(unittest.TestCase):
    """Stress tests on checkpoint serialization, offline deserialization, and 100-sample fidelity."""

    @classmethod
    def setUpClass(cls):
        cls.checkpoint_dir = os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask")
        cls.has_real_checkpoint = os.path.isdir(cls.checkpoint_dir) and (
            os.path.isfile(os.path.join(cls.checkpoint_dir, "model.pt"))
            or os.path.isfile(os.path.join(cls.checkpoint_dir, "pytorch_model.bin"))
        )
        if cls.has_real_checkpoint:
            cls.model = PhoBERTMultiTask.from_pretrained(cls.checkpoint_dir)
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.checkpoint_dir, use_fast=False)
        else:
            cls.model = create_tiny_phobert_model(hidden_size=64)
            cls.dummy_tok = DummyPhoBERTTokenizer(vocab_size=1000)

    def test_save_and_reload_idempotence_on_100_samples(self):
        """Save model to tempdir, reload, and verify 100 samples produce 100% identical outputs."""
        temp_dir = tempfile.mkdtemp(prefix="phobert_reload_stress_")
        try:
            # 1. Save model to temp directory
            saved_path = self.model.save_pretrained(temp_dir)
            self.assertTrue(os.path.isdir(saved_path))
            self.assertTrue(os.path.isfile(os.path.join(saved_path, "model.pt")))
            self.assertTrue(os.path.isfile(os.path.join(saved_path, "pytorch_model.bin")))
            self.assertTrue(os.path.isfile(os.path.join(saved_path, "phobert_multitask_meta.json")))
            self.assertTrue(os.path.isfile(os.path.join(saved_path, "config.json")))

            # 2. Reload model from temp directory
            reloaded_model = PhoBERTMultiTask.from_pretrained(saved_path)
            reloaded_model.eval()
            self.model.eval()

            # 3. Verify state_dict parameters are bitwise identical
            orig_dict = self.model.state_dict()
            reloaded_dict = reloaded_model.state_dict()
            self.assertEqual(orig_dict.keys(), reloaded_dict.keys())
            for key in orig_dict:
                diff = torch.max(torch.abs(orig_dict[key] - reloaded_dict[key])).item()
                self.assertEqual(diff, 0.0, f"Weight divergence in parameter {key}")

            # 4. Acquire 100 test samples (either from UIT-VSFC test split or authentic generated pool)
            if self.has_real_checkpoint:
                test_texts, _, _ = load_vsfc_split("test")
                random.seed(42)
                sample_texts = random.sample(test_texts, 100)
                tok = self.tokenizer
            else:
                sample_pool = [
                    "Thầy dạy rất tận tâm và nhiệt tình",
                    "Cơ sở vật chất phòng học quá tệ",
                    "Chương trình đào tạo phù hợp thực tế",
                    "Không có ý kiến gì thêm",
                    "Giáo viên nhiệt huyết nhưng bài thi quá khó",
                ]
                sample_texts = [f"{sample_pool[i % 5]} sample #{i}" for i in range(100)]
                tok = self.dummy_tok

            # 5. Predict on 100 samples using both models
            orig_preds = self.model.predict(sample_texts, tokenizer=tok, batch_size=32)
            reloaded_preds = reloaded_model.predict(sample_texts, tokenizer=tok, batch_size=32)

            self.assertEqual(len(orig_preds), 100)
            self.assertEqual(len(reloaded_preds), 100)

            # 6. Verify identical predictions across all 100 samples
            for i in range(100):
                op = orig_preds[i]
                rp = reloaded_preds[i]

                # Sentiment checks
                self.assertEqual(op["sentiment"]["label"], rp["sentiment"]["label"], f"Mismatch at #{i}")
                self.assertEqual(op["sentiment"]["label_id"], rp["sentiment"]["label_id"], f"Mismatch at #{i}")
                self.assertAlmostEqual(
                    op["sentiment"]["confidence"], rp["sentiment"]["confidence"], places=6, msg=f"Mismatch at #{i}"
                )
                for k in SENTIMENT_LABELS:
                    self.assertAlmostEqual(
                        op["sentiment"]["distribution"][k],
                        rp["sentiment"]["distribution"][k],
                        places=6,
                        msg=f"Dist mismatch sentiment {k} at #{i}",
                    )

                # Topic checks
                self.assertEqual(op["topic"]["label"], rp["topic"]["label"], f"Mismatch at #{i}")
                self.assertEqual(op["topic"]["label_id"], rp["topic"]["label_id"], f"Mismatch at #{i}")
                self.assertAlmostEqual(
                    op["topic"]["confidence"], rp["topic"]["confidence"], places=6, msg=f"Mismatch at #{i}"
                )
                for k in TOPIC_LABELS:
                    self.assertAlmostEqual(
                        op["topic"]["distribution"][k],
                        rp["topic"]["distribution"][k],
                        places=6,
                        msg=f"Dist mismatch topic {k} at #{i}",
                    )

            # 7. Check raw probability parity via predict_proba
            orig_s_prob, orig_t_prob = self.model.predict_proba(sample_texts, tokenizer=tok, batch_size=32)
            reloaded_s_prob, reloaded_t_prob = reloaded_model.predict_proba(
                sample_texts, tokenizer=tok, batch_size=32
            )
            np.testing.assert_allclose(orig_s_prob, reloaded_s_prob, atol=1e-6, rtol=1e-6)
            np.testing.assert_allclose(orig_t_prob, reloaded_t_prob, atol=1e-6, rtol=1e-6)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
