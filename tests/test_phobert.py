"""
tests/test_phobert.py
Unit Test Suite for PhoBERT Multi-Task Model & Fine-Tuning Pipeline (Milestone 3).

Verifies:
1. Fast Offline/Mock Fixtures: In-memory lightweight model (0 MB download, <0.8s runtime).
2. Model Initialization & Architecture: Submodule dimensions, dropout, device handling.
3. Forward Pass Tensor Shapes: (B, L) -> (B, 3) sentiment logits & (B, 4) topic logits.
4. Multi-Task Joint Loss: CrossEntropy scalar tensor with requires_grad=True & gradient backpropagation.
5. Prediction Method Contract: Strict compliance with PROJECT.md § Interface Contracts.
6. Input Polymorphism & Edge Cases: Single str -> dict, List[str] -> list of dicts, empty/long strings.
7. Checkpoint Save/Load Idempotence: save_pretrained & from_pretrained weight and prediction parity.
8. Dataset & DataLoader Integration: Seamless execution with data/dataset.py (VSFCDataset).

Execution:
    python -m unittest tests/test_phobert.py -v
"""

import math
import os
import shutil
import sys
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Union

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from models.phobert_model import (
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
    SENTIMENT_TO_ID,
    TOPIC_LABELS,
    TOPIC_MAP,
    TOPIC_TO_ID,
    PhoBERTMultiTask,
)


class DummyPhoBERTTokenizer:
    """
    Fast, deterministic offline tokenizer fixture mimicking Hugging Face AutoTokenizer.
    Requires 0 network downloads and enables instantaneous test execution.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 1
        self.cls_token_id = 0
        self.sep_token_id = 2

    def __call__(
        self,
        texts: Union[str, List[str]],
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        if batch_size == 0:
            return {
                "input_ids": torch.empty((0, max_length), dtype=torch.long),
                "attention_mask": torch.empty((0, max_length), dtype=torch.long),
            }

        # Deterministic token IDs based on text hash
        input_ids = torch.ones((batch_size, max_length), dtype=torch.long) * self.pad_token_id
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

        for i, text in enumerate(texts):
            words = text.strip().split()
            seq_len = min(len(words) + 2, max_length)
            input_ids[i, 0] = self.cls_token_id
            for j, word in enumerate(words[: max(0, seq_len - 2)]):
                token_val = (abs(hash(word)) % (self.vocab_size - 10)) + 10
                input_ids[i, j + 1] = token_val
            if seq_len > 1:
                input_ids[i, seq_len - 1] = self.sep_token_id
            attention_mask[i, :seq_len] = 1

        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }


def create_tiny_phobert_model(
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_sentiment: int = 3,
    num_topic: int = 4,
    dropout: float = 0.2,
) -> PhoBERTMultiTask:
    """
    Constructs an in-memory PhoBERTMultiTask instance using a genuine Hugging Face
    RobertaConfig and AutoModel without downloading 540 MB weights.
    """
    conf = AutoConfig.from_pretrained("vinai/phobert-base-v2")
    conf.vocab_size = vocab_size
    conf.hidden_size = hidden_size
    conf.num_hidden_layers = 1
    conf.num_attention_heads = 2
    conf.intermediate_size = hidden_size * 2
    conf.max_position_embeddings = 256

    tiny_backbone = AutoModel.from_config(conf)
    model = PhoBERTMultiTask(
        config=conf,
        backbone=tiny_backbone,
        num_sentiment=num_sentiment,
        num_topic=num_topic,
        dropout=dropout,
    )
    return model


class TestPhoBERTInitialization(unittest.TestCase):
    """Verifies PhoBERTMultiTask initialization, architectural dimensions, and device movement."""

    def test_architecture_dimensions_and_defaults(self):
        model = create_tiny_phobert_model(hidden_size=64, num_sentiment=3, num_topic=4, dropout=0.2)
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.num_sentiment, 3)
        self.assertEqual(model.num_topic, 4)
        self.assertEqual(model.dropout_rate, 0.2)

        # Classifiers verification
        self.assertEqual(model.sentiment_classifier.in_features, 64)
        self.assertEqual(model.sentiment_classifier.out_features, 3)
        self.assertEqual(model.topic_classifier.in_features, 64)
        self.assertEqual(model.topic_classifier.out_features, 4)

        # Initial mode should be train
        self.assertTrue(model.training)

    def test_device_movement(self):
        model = create_tiny_phobert_model(hidden_size=64)
        device = torch.device("cpu")
        model.to(device)
        sample_tensor = torch.zeros((1, 8), dtype=torch.long, device=device)
        output = model(sample_tensor)
        self.assertEqual(output["sentiment_logits"].device.type, "cpu")


class TestPhoBERTForwardPass(unittest.TestCase):
    """Verifies forward pass output tensor shapes, batch sizes, and sequence lengths."""

    @classmethod
    def setUpClass(cls):
        cls.model = create_tiny_phobert_model(hidden_size=64)

    def test_forward_output_shapes_and_keys(self):
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertIsInstance(outputs, dict)
        self.assertIn("sentiment_logits", outputs)
        self.assertIn("topic_logits", outputs)
        self.assertIn("loss", outputs)

        self.assertEqual(outputs["sentiment_logits"].shape, (batch_size, 3))
        self.assertEqual(outputs["topic_logits"].shape, (batch_size, 4))
        self.assertIsNone(outputs["loss"])

    def test_forward_varying_batch_sizes(self):
        for b in [1, 3, 5]:
            input_ids = torch.randint(10, 500, (b, 12), dtype=torch.long)
            outputs = self.model(input_ids=input_ids)
            self.assertEqual(outputs["sentiment_logits"].shape, (b, 3))
            self.assertEqual(outputs["topic_logits"].shape, (b, 4))

    def test_forward_without_attention_mask(self):
        input_ids = torch.randint(10, 500, (2, 8), dtype=torch.long)
        outputs = self.model(input_ids=input_ids, attention_mask=None)
        self.assertEqual(outputs["sentiment_logits"].shape, (2, 3))
        self.assertEqual(outputs["topic_logits"].shape, (2, 4))


class TestPhoBERTLossCalculation(unittest.TestCase):
    """Verifies joint multi-task loss calculation, scalar reduction, and backpropagation."""

    def setUp(self):
        self.model = create_tiny_phobert_model(hidden_size=64)

    def test_loss_scalar_with_labels_and_requires_grad(self):
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        sentiment_labels = torch.tensor([0, 2], dtype=torch.long)
        topic_labels = torch.tensor([1, 3], dtype=torch.long)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentiment_labels=sentiment_labels,
            topic_labels=topic_labels,
        )

        loss = outputs["loss"]
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertTrue(loss.requires_grad)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_loss_gradient_flow_and_backprop(self):
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)
        sentiment_labels = torch.tensor([1, 2], dtype=torch.long)
        topic_labels = torch.tensor([0, 2], dtype=torch.long)

        outputs = self.model(
            input_ids=input_ids,
            sentiment_labels=sentiment_labels,
            topic_labels=topic_labels,
        )
        loss = outputs["loss"]
        loss.backward()

        self.assertIsNotNone(self.model.sentiment_classifier.weight.grad)
        self.assertIsNotNone(self.model.topic_classifier.weight.grad)
        self.assertGreater(self.model.sentiment_classifier.weight.grad.norm().item(), 0.0)
        self.assertGreater(self.model.topic_classifier.weight.grad.norm().item(), 0.0)

    def test_loss_numerical_consistency(self):
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(10, 500, (batch_size, seq_len), dtype=torch.long)
        sentiment_labels = torch.tensor([0, 1], dtype=torch.long)
        topic_labels = torch.tensor([2, 3], dtype=torch.long)

        outputs = self.model(
            input_ids=input_ids,
            sentiment_labels=sentiment_labels,
            topic_labels=topic_labels,
        )
        loss = outputs["loss"]
        sent_logits = outputs["sentiment_logits"]
        topic_logits = outputs["topic_logits"]

        expected_loss = nn.CrossEntropyLoss()(sent_logits, sentiment_labels) + nn.CrossEntropyLoss()(
            topic_logits, topic_labels
        )
        self.assertAlmostEqual(loss.item(), expected_loss.item(), places=5)


class TestPhoBERTPredictionContract(unittest.TestCase):
    """Verifies strict conformance to PROJECT.md § Interface Contracts (Schema, Calibrations & Types)."""

    @classmethod
    def setUpClass(cls):
        cls.model = create_tiny_phobert_model(hidden_size=64)
        cls.tokenizer = DummyPhoBERTTokenizer()

    def test_single_string_prediction_schema(self):
        text = "Thầy giáo dạy rất nhiệt tình và tận tâm"
        pred = self.model.predict(text, tokenizer=self.tokenizer)

        # Must return dictionary for single string
        self.assertIsInstance(pred, dict)
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)

        # Sentiment contract
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

        # Distribution contract
        sent_dist = sent["distribution"]
        self.assertIsInstance(sent_dist, dict)
        self.assertEqual(set(sent_dist.keys()), {"negative", "neutral", "positive"})
        for k, v in sent_dist.items():
            self.assertEqual(type(v), float)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # Probability sum to 1.0
        self.assertTrue(math.isclose(sum(sent_dist.values()), 1.0, rel_tol=1e-5))
        # Confidence must equal predicted label's distribution
        self.assertAlmostEqual(sent["confidence"], sent_dist[sent["label"]], places=5)
        self.assertEqual(sent["label"], SENTIMENT_MAP[sent["label_id"]])

        # Topic contract
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

        # Distribution contract
        topic_dist = topic["distribution"]
        self.assertIsInstance(topic_dist, dict)
        self.assertEqual(set(topic_dist.keys()), {"lecturer", "training_program", "facility", "others"})
        for k, v in topic_dist.items():
            self.assertEqual(type(v), float)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        # Probability sum to 1.0
        self.assertTrue(math.isclose(sum(topic_dist.values()), 1.0, rel_tol=1e-5))
        # Confidence must equal predicted label's distribution
        self.assertAlmostEqual(topic["confidence"], topic_dist[topic["label"]], places=5)
        self.assertEqual(topic["label"], TOPIC_MAP[topic["label_id"]])

    def test_batch_prediction_schema(self):
        batch = [
            "Giảng viên giải thích bài tập rõ ràng",
            "Cơ sở vật chất phòng máy bị hỏng",
            "Nội dung môn học quá nặng",
        ]
        preds = self.model.predict(batch, tokenizer=self.tokenizer)

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


class TestPhoBERTPolymorphismAndEdgeCases(unittest.TestCase):
    """Verifies polymorphic input handling, empty strings, whitespace, and sequence truncations."""

    @classmethod
    def setUpClass(cls):
        cls.model = create_tiny_phobert_model(hidden_size=64)
        cls.tokenizer = DummyPhoBERTTokenizer()

    def test_single_element_list_vs_single_string(self):
        text = "Khóa học rất bổ ích"
        single_pred = self.model.predict(text, tokenizer=self.tokenizer)
        list_pred = self.model.predict([text], tokenizer=self.tokenizer)

        self.assertIsInstance(single_pred, dict)
        self.assertIsInstance(list_pred, list)
        self.assertEqual(len(list_pred), 1)
        self.assertEqual(single_pred["sentiment"]["label"], list_pred[0]["sentiment"]["label"])
        self.assertEqual(single_pred["topic"]["label"], list_pred[0]["topic"]["label"])

    def test_empty_list_input(self):
        preds = self.model.predict([], tokenizer=self.tokenizer)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 0)

    def test_empty_string_and_whitespace(self):
        for empty_text in ["", "   ", "\t\n"]:
            pred = self.model.predict(empty_text, tokenizer=self.tokenizer)
            self.assertIsInstance(pred, dict)
            self.assertIn("sentiment", pred)
            self.assertIn("topic", pred)

    def test_long_string_truncation(self):
        long_text = "tuyệt vời " * 300
        pred = self.model.predict(long_text, tokenizer=self.tokenizer, max_length=128)
        self.assertIsInstance(pred, dict)
        self.assertIn("sentiment", pred)
        self.assertIn("topic", pred)


class TestPhoBERTCheckpointSaveLoad(unittest.TestCase):
    """Verifies save_pretrained and from_pretrained checkpoint idempotence and output parity."""

    def test_save_and_from_pretrained_idempotence(self):
        original_model = create_tiny_phobert_model(hidden_size=64)
        tokenizer = DummyPhoBERTTokenizer()
        test_text = "Giảng viên luôn hỗ trợ sinh viên kịp thời"

        # Obtain pre-save outputs
        pred_before = original_model.predict(test_text, tokenizer=tokenizer)
        sample_input = tokenizer(test_text, max_length=16, return_tensors="pt")
        original_model.eval()
        with torch.no_grad():
            logits_before = original_model(input_ids=sample_input["input_ids"])

        with tempfile.TemporaryDirectory() as temp_dir:
            original_model.save_pretrained(temp_dir)

            # Verify files exist on disk
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))
            self.assertTrue(
                os.path.exists(os.path.join(temp_dir, "pytorch_model.bin"))
                or os.path.exists(os.path.join(temp_dir, "model.pt"))
            )

            # Reload model
            reloaded_model = PhoBERTMultiTask.from_pretrained(temp_dir)
            reloaded_model.eval()

            with torch.no_grad():
                logits_after = reloaded_model(input_ids=sample_input["input_ids"])

            # Verify logit parity
            torch.testing.assert_close(
                logits_before["sentiment_logits"],
                logits_after["sentiment_logits"],
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                logits_before["topic_logits"],
                logits_after["topic_logits"],
                rtol=1e-5,
                atol=1e-5,
            )

            # Verify prediction schema parity
            pred_after = reloaded_model.predict(test_text, tokenizer=tokenizer)
            self.assertEqual(pred_before["sentiment"]["label"], pred_after["sentiment"]["label"])
            self.assertEqual(pred_before["topic"]["label"], pred_after["topic"]["label"])
            self.assertAlmostEqual(
                pred_before["sentiment"]["confidence"],
                pred_after["sentiment"]["confidence"],
                places=5,
            )


class TestPhoBERTDatasetAndDataLoaderIntegration(unittest.TestCase):
    """Verifies integration with data.dataset.VSFCDataset and DataLoader single training step."""

    def test_vsfc_dataset_batch_forward_and_optimizer_step(self):
        from data.dataset import VSFCDataset
        from torch.utils.data import DataLoader

        model = create_tiny_phobert_model(hidden_size=64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        dummy_ids = torch.randint(10, 500, (4, 16), dtype=torch.long)
        dummy_mask = torch.ones((4, 16), dtype=torch.long)
        dummy_sents = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        dummy_topics = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        dataset = VSFCDataset(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            sentiment_labels=dummy_sents,
            topic_labels=dummy_topics,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        initial_weight = model.sentiment_classifier.weight.clone()

        for batch in loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                sentiment_labels=batch["sentiment_labels"],
                topic_labels=batch["topic_labels"],
            )
            loss = outputs["loss"]
            self.assertIsNotNone(loss)
            loss.backward()
            optimizer.step()
            break

        # Check weights updated
        self.assertFalse(torch.equal(model.sentiment_classifier.weight, initial_weight))


class TestPhoBERTCachedRealTokenizer(unittest.TestCase):
    """Verifies integration with actual local cached vinai/phobert-base-v2 tokenizer."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        try:
            cls.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
            cls.model = create_tiny_phobert_model(vocab_size=len(cls.tokenizer), hidden_size=64)
            cls.available = True
        except Exception:
            cls.available = False

    def test_real_tokenizer_prediction(self):
        if not self.available:
            self.skipTest("vinai/phobert-base-v2 tokenizer not available")

        vietnamese_text = "Giảng viên nhiệt tình, tài liệu dễ hiểu."
        pred = self.model.predict(vietnamese_text, tokenizer=self.tokenizer)

        self.assertIsInstance(pred, dict)
        self.assertIn(pred["sentiment"]["label"], SENTIMENT_LABELS)
        self.assertIn(pred["topic"]["label"], TOPIC_LABELS)
        self.assertAlmostEqual(sum(pred["sentiment"]["distribution"].values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
