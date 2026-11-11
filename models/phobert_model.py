"""
models/phobert_model.py
PhoBERT Multi-Task Classification Model for Vietnamese Student Feedback.
Shared pre-trained PhoBERT backbone with dual linear classification heads for Sentiment and Topic.
Strictly conforms to PROJECT.md § Interface Contracts.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Official UIT-VSFC Label Mappings conforming to data/dataset.py & PROJECT.md
SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]
TOPIC_LABELS: List[str] = ["lecturer", "training_program", "facility", "others"]

SENTIMENT_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_MAP: Dict[int, str] = {0: "lecturer", 1: "training_program", 2: "facility", 3: "others"}

SENTIMENT_TO_ID: Dict[str, int] = {v: k for k, v in SENTIMENT_MAP.items()}
TOPIC_TO_ID: Dict[str, int] = {v: k for k, v in TOPIC_MAP.items()}


def _segment_text(text: Optional[str]) -> str:
    """Applies PyVi word segmentation idempotently."""
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(cleaned)
    except ImportError:
        return cleaned


class PhoBERTMultiTask(nn.Module):
    """
    Multi-Task Transformer Model for Vietnamese Student Feedback Classification.

    Combines a shared vinai/phobert-base-v2 backbone with dual linear heads:
    - Sentiment Head: 3 classes (negative, neutral, positive)
    - Topic Head: 4 classes (lecturer, training_program, facility, others)
    """

    def __init__(
        self,
        model_name_or_path: str = "vinai/phobert-base-v2",
        num_sentiment_classes: int = 3,
        num_topic_classes: int = 4,
        dropout_rate: float = 0.2,
        config: Optional[AutoConfig] = None,
        backbone: Optional[nn.Module] = None,
        num_sentiment: Optional[int] = None,
        num_topic: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        # Handle parameter aliases for maximum interoperability
        if num_sentiment is not None:
            num_sentiment_classes = num_sentiment
        if num_topic is not None:
            num_topic_classes = num_topic
        if dropout is not None:
            dropout_rate = dropout

        self.model_name_or_path = model_name_or_path
        self.num_sentiment = num_sentiment_classes
        self.num_sentiment_classes = num_sentiment_classes
        self.num_topic = num_topic_classes
        self.num_topic_classes = num_topic_classes
        self.dropout_rate = dropout_rate

        # 1. Config initialization
        if config is not None:
            self.config = config
        else:
            self.config = AutoConfig.from_pretrained(model_name_or_path)

        # 2. Backbone initialization
        if backbone is not None:
            self.phobert = backbone
        else:
            self.phobert = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        # 3. Regularization & Multi-Task Heads
        hidden_size = getattr(self.config, "hidden_size", 768)
        self.dropout = nn.Dropout(dropout_rate)
        self.sentiment_classifier = nn.Linear(hidden_size, num_sentiment_classes)
        self.topic_classifier = nn.Linear(hidden_size, num_topic_classes)

        self.sentiment_labels = list(SENTIMENT_LABELS)
        self.topic_labels = list(TOPIC_LABELS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        topic_labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass through shared PhoBERT encoder and dual classification heads.

        Args:
            input_ids: LongTensor of token IDs, shape (batch_size, seq_len).
            attention_mask: LongTensor of attention mask, shape (batch_size, seq_len).
            sentiment_labels: Optional LongTensor of sentiment targets, shape (batch_size,).
            topic_labels: Optional LongTensor of topic targets, shape (batch_size,).
            token_type_ids: Optional token type ids for compatibility.

        Returns:
            Dictionary containing:
                'loss': Joint cross-entropy loss (or None if no labels provided),
                'sentiment_logits': Logits of shape (batch_size, num_sentiment_classes),
                'topic_logits': Logits of shape (batch_size, num_topic_classes).
        """
        extra_kwargs = {}
        if token_type_ids is not None:
            extra_kwargs["token_type_ids"] = token_type_ids

        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **extra_kwargs,
        )

        # Extract pooled CLS representation at index 0 (<s> token in RoBERTa)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        # Dual head linear projections
        sentiment_logits = self.sentiment_classifier(cls_rep)
        topic_logits = self.topic_classifier(cls_rep)

        loss = None
        loss_fn = nn.CrossEntropyLoss()
        if sentiment_labels is not None and topic_labels is not None:
            loss_sent = loss_fn(sentiment_logits, sentiment_labels.long())
            loss_topic = loss_fn(topic_logits, topic_labels.long())
            loss = loss_sent + loss_topic
        elif sentiment_labels is not None:
            loss = loss_fn(sentiment_logits, sentiment_labels.long())
        elif topic_labels is not None:
            loss = loss_fn(topic_logits, topic_labels.long())

        return {
            "loss": loss,
            "sentiment_logits": sentiment_logits,
            "topic_logits": topic_logits,
        }

    def predict_proba(
        self,
        texts: Union[str, List[str]],
        tokenizer: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
        batch_size: int = 32,
        word_segment: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes calibrated class probabilities for sentiment and topic.

        Args:
            texts: Single string or list of Vietnamese feedback sentences.
            tokenizer: Hugging Face AutoTokenizer for PhoBERT (or DummyPhoBERTTokenizer).
            device: Optional torch device. Defaults to model's current device.
            max_length: Maximum sequence length for truncation/padding.
            batch_size: Mini-batch size for processing multiple texts safely.
            word_segment: If True, applies PyVi word segmentation before tokenization.

        Returns:
            Tuple of (sentiment_probs, topic_probs) as NumPy arrays.
        """
        if isinstance(texts, str):
            input_list = [texts]
        else:
            input_list = list(texts)

        if len(input_list) == 0:
            return (
                np.empty((0, self.num_sentiment_classes), dtype=np.float64),
                np.empty((0, self.num_topic_classes), dtype=np.float64),
            )

        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)

        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        was_training = self.training
        self.eval()

        all_sent_probs: List[np.ndarray] = []
        all_topic_probs: List[np.ndarray] = []

        try:
            with torch.no_grad():
                for start_idx in range(0, len(input_list), batch_size):
                    batch_texts = input_list[start_idx : start_idx + batch_size]
                    processed = [_segment_text(t) if word_segment else str(t or "") for t in batch_texts]
                    encoded = tokenizer(
                        processed,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    batch_input_ids = encoded["input_ids"].to(device)
                    batch_attention_mask = encoded.get("attention_mask")
                    if batch_attention_mask is not None:
                        batch_attention_mask = batch_attention_mask.to(device)

                    token_type_ids = encoded.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device)

                    outputs = self.forward(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    s_probs = (
                        torch.softmax(outputs["sentiment_logits"], dim=-1)
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    t_probs = (
                        torch.softmax(outputs["topic_logits"], dim=-1)
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                    all_sent_probs.append(s_probs)
                    all_topic_probs.append(t_probs)
        finally:
            if was_training:
                self.train()

        return np.vstack(all_sent_probs), np.vstack(all_topic_probs)

    def predict_labels(
        self,
        texts: Union[str, List[str]],
        tokenizer: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
        batch_size: int = 32,
        word_segment: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Fast vectorized helper returning predicted label IDs.

        Returns:
            Tuple of (sentiment_label_ids, topic_label_ids) as integer lists.
        """
        sent_probs, topic_probs = self.predict_proba(
            texts=texts,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            word_segment=word_segment,
        )
        if len(sent_probs) == 0:
            return [], []
        sent_ids = [int(x) for x in np.argmax(sent_probs, axis=1)]
        topic_ids = [int(x) for x in np.argmax(topic_probs, axis=1)]
        return sent_ids, topic_ids

    def predict(
        self,
        texts: Union[str, List[str]],
        tokenizer: Any = None,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
        batch_size: int = 32,
        word_segment: bool = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Unified prediction interface conforming strictly to PROJECT.md § Interface Contracts.

        Args:
            texts: Single string or list of Vietnamese feedback sentences.
            tokenizer: Hugging Face AutoTokenizer for PhoBERT (or DummyPhoBERTTokenizer).
            device: Optional torch device.
            max_length: Maximum sequence length.
            batch_size: Processing batch size.
            word_segment: If True, applies PyVi segmentation.

        Returns:
            If single string: A single dictionary with 'sentiment' and 'topic' fields.
            If list: A list of dictionaries matching the same contract.
        """
        is_single = isinstance(texts, str)
        if is_single:
            input_list = [texts]
        else:
            input_list = list(texts)

        if len(input_list) == 0:
            return []

        sent_probs, topic_probs = self.predict_proba(
            texts=input_list,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            word_segment=word_segment,
        )

        results: List[Dict[str, Any]] = []
        for i in range(len(input_list)):
            s_row = sent_probs[i]
            t_row = topic_probs[i]

            # Re-normalize to avoid floating point summation inaccuracies
            s_sum = float(np.sum(s_row))
            if s_sum > 0:
                s_row = s_row / s_sum
            t_sum = float(np.sum(t_row))
            if t_sum > 0:
                t_row = t_row / t_sum

            s_id = int(np.argmax(s_row))
            t_id = int(np.argmax(t_row))

            s_dist = {SENTIMENT_MAP[c]: float(s_row[c]) for c in range(self.num_sentiment_classes)}
            t_dist = {TOPIC_MAP[c]: float(t_row[c]) for c in range(self.num_topic_classes)}

            s_conf = float(s_dist[SENTIMENT_MAP[s_id]])
            t_conf = float(t_dist[TOPIC_MAP[t_id]])

            item = {
                "sentiment": {
                    "label": SENTIMENT_MAP[s_id],
                    "label_id": s_id,
                    "confidence": s_conf,
                    "distribution": s_dist,
                },
                "topic": {
                    "label": TOPIC_MAP[t_id],
                    "label_id": t_id,
                    "confidence": t_conf,
                    "distribution": t_dist,
                },
            }
            results.append(item)

        return results[0] if is_single else results

    def save_pretrained(self, save_directory: Union[str, os.PathLike]) -> str:
        """
        Serializes model weights and configuration to a directory.

        Args:
            save_directory: Directory path where weights and config will be stored.

        Returns:
            Absolute path to save_directory.
        """
        save_dir = os.path.abspath(save_directory)
        os.makedirs(save_dir, exist_ok=True)

        # 1. Save Hugging Face config
        if hasattr(self.config, "save_pretrained"):
            self.config.architectures = ["PhoBERTMultiTask"]
            self.config.num_sentiment_classes = self.num_sentiment_classes
            self.config.num_topic_classes = self.num_topic_classes
            self.config.dropout_rate = self.dropout_rate
            self.config.save_pretrained(save_dir)

        # 2. Save metadata JSON
        meta = {
            "model_type": "PhoBERTMultiTask",
            "base_model_name_or_path": self.model_name_or_path,
            "num_sentiment_classes": self.num_sentiment_classes,
            "num_topic_classes": self.num_topic_classes,
            "dropout_rate": self.dropout_rate,
            "hidden_size": getattr(self.config, "hidden_size", 768),
        }
        with open(os.path.join(save_dir, "phobert_multitask_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 3. Save weights state dict
        weights_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), weights_path)
        # Also save pytorch_model.bin for standard HF tooling compatibility
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

        return save_dir

    @classmethod
    def from_pretrained(
        cls,
        save_directory: Union[str, os.PathLike],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "PhoBERTMultiTask":
        """
        Loads a PhoBERTMultiTask instance from a local checkpoint directory or HF Hub model name.

        Args:
            save_directory: Local checkpoint path or Hugging Face model identifier.
            device: Optional torch device to move the loaded model to.
            **kwargs: Extra parameters passed to model initialization.

        Returns:
            Instantiated and loaded PhoBERTMultiTask model.
        """
        save_dir_str = str(save_directory)
        is_local = os.path.isdir(save_dir_str)

        if is_local:
            pt_file = os.path.join(save_dir_str, "model.pt")
            bin_file = os.path.join(save_dir_str, "pytorch_model.bin")
            weights_file = pt_file if os.path.isfile(pt_file) else bin_file
            if not os.path.isfile(weights_file):
                raise FileNotFoundError(
                    f"No checkpoint weights (model.pt or pytorch_model.bin) found in '{save_dir_str}'"
                )

            # Read metadata if present
            meta_file = os.path.join(save_dir_str, "phobert_multitask_meta.json")
            if os.path.isfile(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                num_sentiment = meta.get("num_sentiment_classes", kwargs.get("num_sentiment_classes", 3))
                num_topic = meta.get("num_topic_classes", kwargs.get("num_topic_classes", 4))
                dropout_rate = meta.get("dropout_rate", kwargs.get("dropout_rate", 0.2))
                base_model = meta.get(
                    "base_model_name_or_path",
                    kwargs.get("model_name_or_path", "vinai/phobert-base-v2"),
                )
            else:
                num_sentiment = kwargs.get("num_sentiment_classes", kwargs.get("num_sentiment", 3))
                num_topic = kwargs.get("num_topic_classes", kwargs.get("num_topic", 4))
                dropout_rate = kwargs.get("dropout_rate", kwargs.get("dropout", 0.2))
                base_model = kwargs.get("model_name_or_path", "vinai/phobert-base-v2")

            config_file = os.path.join(save_dir_str, "config.json")
            if os.path.isfile(config_file):
                config = AutoConfig.from_pretrained(save_dir_str)
            else:
                config = AutoConfig.from_pretrained(base_model)

            # Instantiate backbone architecture without network download
            try:
                backbone = AutoModel.from_config(config)
            except Exception:
                backbone = AutoModel.from_pretrained(base_model, config=config)

            model = cls(
                model_name_or_path=base_model,
                num_sentiment_classes=num_sentiment,
                num_topic_classes=num_topic,
                dropout_rate=dropout_rate,
                config=config,
                backbone=backbone,
            )

            map_location = device if device is not None else "cpu"
            try:
                state_dict = torch.load(weights_file, map_location=map_location, weights_only=True)
            except Exception:
                state_dict = torch.load(weights_file, map_location=map_location)

            model.load_state_dict(state_dict)

            if device is not None:
                model.to(device)

            return model
        else:
            # Remote Hugging Face model identifier
            config = AutoConfig.from_pretrained(save_dir_str)
            model = cls(
                model_name_or_path=save_dir_str,
                config=config,
                **kwargs,
            )
            if device is not None:
                model.to(device)
            return model
