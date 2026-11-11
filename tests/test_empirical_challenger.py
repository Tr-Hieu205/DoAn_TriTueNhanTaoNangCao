"""
tests/test_empirical_challenger.py
Empirical Adversarial & Stress Test Suite for Milestone 1: Data Acquisition & Preprocessing Pipeline.

Author: Challenger M1_1 (critic, specialist)
Validates:
  1. Genuine reading, tokenization, and batching of UIT-VSFC parquet splits
  2. Integration with official vinai/phobert-base-v2 tokenizer and PyVi word segmentation
  3. Custom batch sizes (1, 2, 7, 64, 128, 500) and boundary sequence lengths (8, 64, 128, 256)
  4. Extreme text inputs (empty string, whitespace, 5000-word spam, emojis, special characters, anonymized tokens)
  5. Missing files, corrupted parquet files, schema violations, out-of-bounds labels
  6. Inference mode without labels (unlabeled dataset support)
"""

import os
import sys
import shutil
import tempfile
import unittest
from typing import List, Dict, Any

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoTokenizer
from pyvi import ViTokenizer

from data.download_data import download_vsfc_dataset, EXPECTED_ROW_COUNTS
from data.dataset import (
    load_vsfc_split,
    VSFCDataset,
    create_dataloader,
    get_dataloaders,
    _read_table_file,
    SENTIMENT_MAP,
    TOPIC_MAP,
)


class TestGenuineDataPipeline(unittest.TestCase):
    """Suite 1: Empirical verification that pipeline genuinely reads, tokenizes, and batches UIT-VSFC."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(PROJECT_ROOT, "data", "uit_vsfc")
        # Ensure tokenizer is loaded
        cls.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

    def test_genuine_parquet_splits_exist_and_match_specs(self):
        """Verify train, validation, test parquets exist, have correct magic header, and expected sizes."""
        for split, expected_count in EXPECTED_ROW_COUNTS.items():
            path = os.path.join(self.data_dir, f"{split}.parquet")
            self.assertTrue(os.path.exists(path), f"Parquet file missing: {path}")
            size = os.path.getsize(path)
            self.assertGreater(size, 50000, f"Parquet file {path} size unexpectedly small: {size} bytes")

            # Validate PAR1 header
            with open(path, "rb") as f:
                header = f.read(4)
                self.assertEqual(header, b"PAR1", f"Parquet magic bytes missing in {path}")

            # Validate loaded row counts
            texts, sents, topics = load_vsfc_split(split, data_dir=self.data_dir, word_segment=False)
            self.assertEqual(len(texts), expected_count)
            self.assertEqual(len(sents), expected_count)
            self.assertEqual(len(topics), expected_count)

    def test_genuine_data_content_authenticity(self):
        """Verify that loaded texts contain authentic Vietnamese feedback sentences and labels."""
        train_texts, train_sents, train_topics = load_vsfc_split("train", data_dir=self.data_dir, word_segment=False)

        # Authentic Vietnamese diacritics check
        vietnamese_diacritics = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
        has_diacritics = any(any(c in vietnamese_diacritics for c in text.lower()) for text in train_texts[:100])
        self.assertTrue(has_diacritics, "Dataset does not contain expected Vietnamese diacritics")

        # Check label class distribution completeness
        sent_classes = set(train_sents)
        topic_classes = set(train_topics)
        self.assertEqual(sent_classes, {0, 1, 2}, f"Unexpected sentiment set: {sent_classes}")
        self.assertEqual(topic_classes, {0, 1, 2, 3}, f"Unexpected topic set: {topic_classes}")

    def test_phobert_tokenizer_special_tokens_and_masks(self):
        """Verify real PhoBERT tokenization produces <s> (0), </s> (2), <pad> (1), and matching attention masks."""
        sample_texts = ["thầy dạy rất nhiệt_tình", "cơ_sở vật_chất quá tệ"]
        dataset = VSFCDataset.from_texts(
            texts=sample_texts,
            sentiments=[2, 0],
            topics=[0, 2],
            tokenizer=self.tokenizer,
            max_length=16,
        )

        item0 = dataset[0]
        input_ids = item0["input_ids"]
        attention_mask = item0["attention_mask"]

        # First token should be <s> (id 0 in PhoBERT)
        self.assertEqual(input_ids[0].item(), 0, "BOS token <s> must be at index 0")
        # Non-pad tokens must have attention_mask == 1
        num_tokens = attention_mask.sum().item()
        self.assertGreater(num_tokens, 2, "Should have more than 2 non-pad tokens")
        # Last non-pad token should be </s> (id 2 in PhoBERT)
        self.assertEqual(input_ids[num_tokens - 1].item(), 2, "EOS token </s> must terminate sentence")
        # Padding tokens must be 1 (<pad>) and have attention_mask == 0
        if num_tokens < 16:
            self.assertEqual(input_ids[num_tokens].item(), 1, "Padding token must be 1 (<pad>)")
            self.assertEqual(attention_mask[num_tokens].item(), 0, "Padding mask must be 0")


class TestStressBatchSizesAndSequenceLengths(unittest.TestCase):
    """Suite 2: Boundary and stress testing on batch sizes and sequence lengths."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)
        cls.texts, cls.sents, cls.topics = load_vsfc_split("validation", word_segment=False)
        # Use first 50 samples for fast parametrized stress testing
        cls.sample_texts = cls.texts[:50]
        cls.sample_sents = cls.sents[:50]
        cls.sample_topics = cls.topics[:50]

    def test_batch_sizes(self):
        """Stress-test custom batch sizes: 1, 2, 7 (prime), 64, 128, 500 (> total items)."""
        batch_sizes = [1, 2, 7, 64, 128, 500]

        for bs in batch_sizes:
            loader = create_dataloader(
                texts=self.sample_texts,
                sentiments=self.sample_sents,
                topics=self.sample_topics,
                tokenizer=self.tokenizer,
                batch_size=bs,
                max_length=32,
                shuffle=False,
            )

            total_samples = 0
            for batch in loader:
                b_size = batch["input_ids"].shape[0]
                self.assertLessEqual(b_size, bs)
                self.assertEqual(batch["input_ids"].shape, torch.Size([b_size, 32]))
                self.assertEqual(batch["attention_mask"].shape, torch.Size([b_size, 32]))
                self.assertEqual(batch["sentiment_labels"].shape, torch.Size([b_size]))
                self.assertEqual(batch["topic_labels"].shape, torch.Size([b_size]))
                total_samples += b_size

            self.assertEqual(total_samples, 50, f"Batch size {bs} failed to yield all 50 samples")

    def test_sequence_lengths(self):
        """Stress-test sequence lengths: 8 (ultra short), 64 (medium), 128 (standard), 256 (long)."""
        lengths = [8, 64, 128, 256]

        for max_len in lengths:
            loader = create_dataloader(
                texts=self.sample_texts[:10],
                sentiments=self.sample_sents[:10],
                topics=self.sample_topics[:10],
                tokenizer=self.tokenizer,
                batch_size=4,
                max_length=max_len,
            )
            first_batch = next(iter(loader))
            self.assertEqual(first_batch["input_ids"].shape[1], max_len)
            self.assertEqual(first_batch["attention_mask"].shape[1], max_len)


class TestExtremeInputsAndRobustness(unittest.TestCase):
    """Suite 3: Stress-testing adversarial and extreme text inputs."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

    def test_empty_and_whitespace_strings(self):
        """Verify empty string and whitespace strings do not cause crash."""
        adversarial_texts = ["", "   ", "\t\t\n\n", "    "]
        dataset = VSFCDataset.from_texts(
            texts=adversarial_texts,
            sentiments=[1, 1, 1, 1],
            topics=[3, 3, 3, 3],
            tokenizer=self.tokenizer,
            max_length=16,
        )
        self.assertEqual(len(dataset), 4)
        for i in range(4):
            item = dataset[i]
            # Even for empty text, BOS and EOS are added
            self.assertEqual(item["input_ids"].shape, torch.Size([16]))
            self.assertEqual(item["attention_mask"].shape, torch.Size([16]))
            self.assertGreaterEqual(item["attention_mask"].sum().item(), 2)

    def test_extreme_length_string_truncation(self):
        """Verify 5,000-word paragraph is truncated cleanly without MemoryError or crash."""
        huge_sentence = "giáo viên rất nhiệt tình và tận tâm giảng dạy sinh viên " * 500  # ~4,500 words
        dataset = VSFCDataset.from_texts(
            texts=[huge_sentence],
            sentiments=[2],
            topics=[0],
            tokenizer=self.tokenizer,
            max_length=128,
        )
        item = dataset[0]
        self.assertEqual(item["input_ids"].shape, torch.Size([128]))
        self.assertEqual(item["attention_mask"].sum().item(), 128)

    def test_emojis_special_characters_anonymized_placeholders(self):
        """Verify emojis, weird punctuation, HTML tags, and anonymized tokens do not break segmentation or tokenization."""
        weird_texts = [
            "😊 🚀 👍 Thầy dạy quá đỉnh :))) <3",
            "Phòng học nóng như lò thiêu @#$%^&*(!)",
            "<p>Học phí tăng <b>quá nhiều</b></p>",
            "thầy wzjwz208 dạy môn wzjwz14 điểm thi 10/10",
            "12345 67890 3.1415926535",
        ]

        # 1. PyVi segmentation
        segmented = [ViTokenizer.tokenize(t) for t in weird_texts]
        self.assertEqual(len(segmented), 5)

        # 2. PhoBERT tokenization
        dataset = VSFCDataset.from_texts(
            texts=segmented,
            sentiments=[2, 0, 0, 2, 1],
            topics=[0, 2, 1, 0, 3],
            tokenizer=self.tokenizer,
            max_length=64,
        )
        self.assertEqual(len(dataset), 5)
        for i in range(5):
            item = dataset[i]
            self.assertEqual(item["input_ids"].shape, torch.Size([64]))


class TestErrorHandlingAndCorruptInputs(unittest.TestCase):
    """Suite 4: Corrupt inputs, missing files, invalid labels, and exception handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="uit_vsfc_adversarial_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_split_name_raises(self):
        """Verify invalid split names raise ValueError."""
        for invalid_name in ["unknown", "eval", "testing123", "", "   "]:
            with self.assertRaises(ValueError):
                load_vsfc_split(invalid_name, data_dir="data/uit_vsfc")

    def test_missing_split_file_raises_filenotfound(self):
        """Verify missing split file with auto_download=False raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_vsfc_split("train", data_dir=self.temp_dir, auto_download=False)

    def test_corrupted_parquet_file_raises(self):
        """Verify loading a corrupt/truncated parquet file raises error."""
        corrupt_file = os.path.join(self.temp_dir, "train.parquet")
        with open(corrupt_file, "wb") as f:
            f.write(b"NOT_A_PARQUET_FILE_GARBAGE_BYTES_1234567890")

        with self.assertRaises(Exception):
            load_vsfc_split("train", data_dir=self.temp_dir, auto_download=False)

    def test_missing_required_columns_raises_value_error(self):
        """Verify parquet with missing columns raises ValueError."""
        try:
            import polars as pl
            # Create parquet with missing 'topic' column
            df = pl.DataFrame({
                "sentence": ["Bài học hay", "Thầy giảng tốt"],
                "sentiment": [2, 2],
            })
            bad_parquet = os.path.join(self.temp_dir, "validation.parquet")
            df.write_parquet(bad_parquet)

            with self.assertRaises(ValueError) as ctx:
                load_vsfc_split("validation", data_dir=self.temp_dir, auto_download=False)
            self.assertIn("missing required columns", str(ctx.exception).lower())
        except ImportError:
            self.skipTest("polars not available")

    def test_out_of_bound_sentiment_label_raises_value_error(self):
        """Verify sentiment label outside [0, 2] raises descriptive ValueError."""
        try:
            import polars as pl
            df = pl.DataFrame({
                "sentence": ["Bài học hay"],
                "sentiment": [9],  # Out of bounds!
                "topic": [0],
            })
            bad_parquet = os.path.join(self.temp_dir, "test.parquet")
            df.write_parquet(bad_parquet)

            with self.assertRaises(ValueError) as ctx:
                load_vsfc_split("test", data_dir=self.temp_dir, auto_download=False)
            self.assertIn("invalid sentiment label", str(ctx.exception).lower())
        except ImportError:
            self.skipTest("polars not available")

    def test_out_of_bound_topic_label_raises_value_error(self):
        """Verify topic label outside [0, 3] raises descriptive ValueError."""
        try:
            import polars as pl
            df = pl.DataFrame({
                "sentence": ["Cơ sở vật chất tốt"],
                "sentiment": [2],
                "topic": [5],  # Out of bounds!
                "topic": [5],
            })
            bad_parquet = os.path.join(self.temp_dir, "test.parquet")
            df.write_parquet(bad_parquet)

            with self.assertRaises(ValueError) as ctx:
                load_vsfc_split("test", data_dir=self.temp_dir, auto_download=False)
            self.assertIn("invalid topic label", str(ctx.exception).lower())
        except ImportError:
            self.skipTest("polars not available")


class TestInferenceModeWithoutLabels(unittest.TestCase):
    """Suite 5: Verifies inference/production mode where labels are not provided."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

    def test_unlabeled_dataset_and_dataloader(self):
        """Verify VSFCDataset and DataLoader can ingest raw unlabelled text for inference."""
        inference_texts = [
            "Giảng viên giảng bài rất dễ hiểu và truyền cảm hứng",
            "Máy lạnh phòng C301 bị hỏng kêu to quá",
        ]

        # Dataset without sentiments or topics
        dataset = VSFCDataset.from_texts(
            texts=inference_texts,
            tokenizer=self.tokenizer,
            max_length=64,
        )
        self.assertEqual(len(dataset), 2)

        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertNotIn("sentiment_labels", item)
        self.assertNotIn("topic_labels", item)

        loader = create_dataloader(
            texts=inference_texts,
            tokenizer=self.tokenizer,
            batch_size=2,
            max_length=64,
        )
        batch = next(iter(loader))
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertNotIn("sentiment_labels", batch)
        self.assertNotIn("topic_labels", batch)
        self.assertEqual(batch["input_ids"].shape, torch.Size([2, 64]))


if __name__ == "__main__":
    unittest.main()
