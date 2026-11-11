"""
tests/test_data_pipeline.py
Automated verification test suite for Milestone 1 (Data Acquisition & Preprocessing Pipeline).

Execution:
    python -m unittest tests/test_data_pipeline.py -v
    python tests/test_data_pipeline.py
"""

import os
import sys
import unittest
from typing import Dict, List, Any

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from data.download_data import download_vsfc_dataset, EXPECTED_ROW_COUNTS
from data.dataset import load_vsfc_split, VSFCDataset, get_dataloaders


class DummyPhoBERTTokenizer:
    """
    Lightweight mock tokenizer mimicking Hugging Face AutoTokenizer.
    Permits deterministic, offline verification of VSFCDataset and DataLoader batch tensor shapes.
    """
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(
        self,
        texts: List[str],
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        batch_size = len(texts)
        # Create deterministic pseudo input IDs
        input_ids = torch.randint(low=1, high=self.vocab_size, size=(batch_size, max_length), dtype=torch.long)
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


class TestDataAcquisition(unittest.TestCase):
    """Verifies automated download, caching, and file integrity of UIT-VSFC parquet splits."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(PROJECT_ROOT, "data", "uit_vsfc")
        cls.results = download_vsfc_dataset(output_dir=cls.data_dir, force=False)

    def test_cache_directory_exists(self):
        """Assertion A01: Cache directory is created."""
        self.assertTrue(os.path.isdir(self.data_dir), f"Directory {self.data_dir} does not exist")

    def test_parquet_files_exist_and_valid_header(self):
        """Assertions A02-A04: Parquet files exist, non-empty, and contain PAR1 magic header."""
        for split in ["train", "validation", "test"]:
            file_path = os.path.join(self.data_dir, f"{split}.parquet")
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file_path}")
            size = os.path.getsize(file_path)
            self.assertGreater(size, 10000, f"File {file_path} suspiciously small: {size} bytes")

            with open(file_path, "rb") as f:
                header = f.read(4)
                self.assertEqual(header, b"PAR1", f"File {file_path} does not start with Parquet magic bytes b'PAR1'")

    def test_download_caching_idempotence(self):
        """Assertion A05: Second invocation uses local cache without error."""
        cached_results = download_vsfc_dataset(output_dir=self.data_dir, force=False)
        for split in ["train", "validation", "test"]:
            self.assertIn(split, cached_results)
            self.assertTrue(os.path.exists(cached_results[split]))


class TestDatasetSplitsAndSchema(unittest.TestCase):
    """Verifies academic split row counts, schema types, zero missing values, and label domains."""

    def test_exact_split_row_counts(self):
        """Assertions A06-A09: Exact official academic benchmark row counts."""
        expected_counts = {"train": 11426, "validation": 1583, "test": 3166}
        total_rows = 0

        for split, expected in expected_counts.items():
            texts, sentiments, topics = load_vsfc_split(split, word_segment=False)
            self.assertEqual(len(texts), expected, f"Split '{split}' row count {len(texts)} != expected {expected}")
            self.assertEqual(len(sentiments), expected, f"Split '{split}' sentiment count mismatch")
            self.assertEqual(len(topics), expected, f"Split '{split}' topic count mismatch")
            total_rows += len(texts)

        self.assertEqual(total_rows, 16175, f"Total dataset size {total_rows} != 16,175")

    def test_zero_null_or_empty_values(self):
        """Assertions A10-A11: Zero null, NaN, or whitespace-only sentences."""
        for split in ["train", "validation", "test"]:
            texts, sentiments, topics = load_vsfc_split(split, word_segment=False)
            for idx, text in enumerate(texts):
                self.assertIsNotNone(text, f"Null text at {split}[{idx}]")
                self.assertIsInstance(text, str, f"Non-string text at {split}[{idx}]")
                self.assertGreater(len(text.strip()), 0, f"Empty text string at {split}[{idx}]")

            for idx, s in enumerate(sentiments):
                self.assertIsNotNone(s, f"Null sentiment at {split}[{idx}]")
                self.assertIsInstance(s, int, f"Non-integer sentiment at {split}[{idx}]")

            for idx, t in enumerate(topics):
                self.assertIsNotNone(t, f"Null topic at {split}[{idx}]")
                self.assertIsInstance(t, int, f"Non-integer topic at {split}[{idx}]")

    def test_label_value_ranges(self):
        """Assertions A12-A13: Sentiments strictly in {0, 1, 2}, Topics strictly in {0, 1, 2, 3}."""
        for split in ["train", "validation", "test"]:
            _, sentiments, topics = load_vsfc_split(split, word_segment=False)
            sent_set = set(sentiments)
            topic_set = set(topics)

            self.assertEqual(sent_set, {0, 1, 2}, f"Split '{split}' sentiment labels {sent_set} != {{0, 1, 2}}")
            self.assertEqual(topic_set, {0, 1, 2, 3}, f"Split '{split}' topic labels {topic_set} != {{0, 1, 2, 3}}")

    def test_no_cross_split_leakage(self):
        """Assertion A14: No overlapping sentences across splits (data leakage check)."""
        train_texts, _, _ = load_vsfc_split("train", word_segment=False)
        val_texts, _, _ = load_vsfc_split("validation", word_segment=False)
        test_texts, _, _ = load_vsfc_split("test", word_segment=False)

        train_set = set(s.strip().lower() for s in train_texts)
        val_set = set(s.strip().lower() for s in val_texts)
        test_set = set(s.strip().lower() for s in test_texts)

        self.assertEqual(len(train_set & val_set), 0, "Data leakage detected between train and validation splits")
        self.assertEqual(len(train_set & test_set), 0, "Data leakage detected between train and test splits")
        self.assertEqual(len(val_set & test_set), 0, "Data leakage detected between validation and test splits")

    def test_invalid_split_raises_value_error(self):
        """Graceful error handling for invalid split names."""
        with self.assertRaises(ValueError):
            load_vsfc_split("non_existent_split")


class TestVietnameseWordSegmentation(unittest.TestCase):
    """Verifies PyVi word segmentation joining compound tokens with underscore."""

    def test_compound_word_joining(self):
        """Assertion A15: Compound words joined by underscore."""
        try:
            from pyvi import ViTokenizer
        except ImportError:
            self.skipTest("pyvi not yet installed in active environment")

        sample = "Giảng viên nhiệt tình hướng dẫn sinh viên làm bài tập"
        tokenized = ViTokenizer.tokenize(sample)
        self.assertIn("Giảng_viên", tokenized)
        self.assertIn("nhiệt_tình", tokenized)
        self.assertIn("sinh_viên", tokenized)
        self.assertIn("bài_tập", tokenized)

    def test_segmentation_toggle(self):
        """Assertion A16: word_segment=True/False toggle functions properly."""
        try:
            import pyvi
        except ImportError:
            self.skipTest("pyvi not yet installed in active environment")

        texts_unseg, _, _ = load_vsfc_split("validation", word_segment=False)
        texts_seg, _, _ = load_vsfc_split("validation", word_segment=True)

        self.assertEqual(len(texts_unseg), len(texts_seg))
        # Ensure segmented text contains underscore joining for compound words
        underscored_count = sum(1 for s in texts_seg if "_" in s)
        self.assertGreater(underscored_count, len(texts_seg) * 0.8, "Expected >80% of sentences to have compound words")

    def test_special_characters_and_anonymized_tokens(self):
        """Preserves anonymized placeholders (wzjwz208) and punctuations."""
        try:
            from pyvi import ViTokenizer
        except ImportError:
            self.skipTest("pyvi not yet installed in active environment")

        sample = "Thầy wzjwz208 dạy rất hay :))) !!!"
        tokenized = ViTokenizer.tokenize(sample)
        self.assertIn("wzjwz208", tokenized)
        self.assertGreater(len(tokenized), 0)


class TestVSFCDatasetAndDataLoader(unittest.TestCase):
    """Verifies PyTorch Dataset, batch collation, tensor shapes (batch_size, 128), and dtypes."""

    def setUp(self):
        self.dummy_tokenizer = DummyPhoBERTTokenizer()

    def test_dataset_indexing_and_dtypes(self):
        """Assertion A17: Dataset length, dictionary keys, and torch.long dtypes."""
        texts = ["Bài giảng rất hay", "Phòng học quá nóng"]
        sentiments = [2, 0]
        topics = [0, 2]

        dataset = VSFCDataset(
            texts=texts,
            sentiments=sentiments,
            topics=topics,
            tokenizer=self.dummy_tokenizer,
            max_length=128
        )

        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("sentiment_labels", item)
        self.assertIn("topic_labels", item)

        self.assertEqual(item["input_ids"].dtype, torch.long)
        self.assertEqual(item["attention_mask"].dtype, torch.long)
        self.assertEqual(item["sentiment_labels"].dtype, torch.long)
        self.assertEqual(item["topic_labels"].dtype, torch.long)
        self.assertEqual(item["input_ids"].shape, torch.Size([128]))

    def test_dataloader_tensor_shapes(self):
        """Assertions A18-A20: DataLoader yields batches with shapes (batch_size, 128)."""
        batch_size = 16
        max_length = 128

        train_loader, val_loader, test_loader = get_dataloaders(
            tokenizer=self.dummy_tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            quick_mode=True,  # Test on quick subset for instantaneous execution
            data_dir=os.path.join(PROJECT_ROOT, "data", "uit_vsfc")
        )

        first_batch = next(iter(train_loader))
        self.assertEqual(first_batch["input_ids"].shape, torch.Size([batch_size, max_length]))
        self.assertEqual(first_batch["attention_mask"].shape, torch.Size([batch_size, max_length]))
        self.assertEqual(first_batch["sentiment_labels"].shape, torch.Size([batch_size]))
        self.assertEqual(first_batch["topic_labels"].shape, torch.Size([batch_size]))

        self.assertEqual(first_batch["input_ids"].dtype, torch.long)
        self.assertEqual(first_batch["attention_mask"].dtype, torch.long)
        self.assertEqual(first_batch["sentiment_labels"].dtype, torch.long)
        self.assertEqual(first_batch["topic_labels"].dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
