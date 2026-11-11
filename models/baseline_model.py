"""
models/baseline_model.py
Classical Baseline Model for Vietnamese Student Feedback Classification.
Architecture: Shared TF-IDF Vectorizer + Dual Balanced Logistic Regression Heads.
Conforms strictly to PROJECT.md § Interface Contracts.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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


def _calibrate_probs(probs: np.ndarray, classes: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Ensures predicted probability matrix has exactly num_classes columns,
    correctly mapped even if some classes were unobserved in small dummy splits.
    """
    if len(classes) == num_classes and np.array_equal(classes, np.arange(num_classes)):
        return probs
    full_probs = np.zeros((probs.shape[0], num_classes), dtype=np.float64)
    for col_idx, cls_id in enumerate(classes):
        if 0 <= cls_id < num_classes:
            full_probs[:, cls_id] = probs[:, col_idx]
    return full_probs


class VietnameseFeedbackBaseline:
    """
    Classical Baseline Model for Vietnamese Student Feedback Classification.

    Combines a shared TF-IDF feature extractor (n-gram 1-2, sublinear TF) with dual
    balanced Logistic Regression heads for Sentiment (3 classes) and Topic (4 classes).
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.9,
        max_features: int = 10000,
        sublinear_tf: bool = True,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )
        self.sentiment_clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.topic_clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.sentiment_labels = list(SENTIMENT_LABELS)
        self.topic_labels = list(TOPIC_LABELS)
        self.is_fitted = False

    def fit(
        self,
        texts: List[str],
        sentiments: List[int],
        topics: List[int],
        word_segment: bool = True,
    ) -> "VietnameseFeedbackBaseline":
        """
        Fits the shared TF-IDF vectorizer and both Logistic Regression classifiers.

        Args:
            texts: List of Vietnamese feedback strings (raw or pre-segmented).
            sentiments: Integer sentiment labels in {0, 1, 2}.
            topics: Integer topic labels in {0, 1, 2, 3}.
            word_segment: If True, ensures PyVi segmentation is applied.

        Returns:
            self: The fitted model instance.
        """
        if len(texts) != len(sentiments) or len(texts) != len(topics):
            raise ValueError(
                f"Mismatched dataset lengths: texts ({len(texts)}), "
                f"sentiments ({len(sentiments)}), topics ({len(topics)})"
            )

        processed_texts = [_segment_text(t) if word_segment else str(t or "") for t in texts]

        # Shared feature extraction
        X = self.vectorizer.fit_transform(processed_texts)

        # Dual classification heads
        self.sentiment_clf.fit(X, sentiments)
        self.topic_clf.fit(X, topics)

        self.is_fitted = True
        return self

    def predict_proba(
        self, texts: Union[str, List[str]], word_segment: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes calibrated class probabilities for sentiment and topic.

        Returns:
            Tuple of (sentiment_probs, topic_probs) as numpy arrays.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        if isinstance(texts, str):
            input_list = [texts]
        else:
            input_list = list(texts)

        if len(input_list) == 0:
            return (
                np.empty((0, len(SENTIMENT_MAP)), dtype=np.float64),
                np.empty((0, len(TOPIC_MAP)), dtype=np.float64),
            )

        processed = [_segment_text(t) if word_segment else str(t or "") for t in input_list]
        X = self.vectorizer.transform(processed)
        raw_sent_probs = self.sentiment_clf.predict_proba(X)
        raw_topic_probs = self.topic_clf.predict_proba(X)

        sent_probs = _calibrate_probs(raw_sent_probs, self.sentiment_clf.classes_, len(SENTIMENT_MAP))
        topic_probs = _calibrate_probs(raw_topic_probs, self.topic_clf.classes_, len(TOPIC_MAP))
        return sent_probs, topic_probs

    def predict_labels(
        self, texts: Union[str, List[str]], word_segment: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Fast vectorized helper returning predicted label IDs.

        Returns:
            Tuple of (sentiment_label_ids, topic_label_ids) as integer lists.
        """
        sent_probs, topic_probs = self.predict_proba(texts, word_segment=word_segment)
        if len(sent_probs) == 0:
            return [], []
        sent_ids = np.argmax(sent_probs, axis=1).astype(int).tolist()
        topic_ids = np.argmax(topic_probs, axis=1).astype(int).tolist()
        return sent_ids, topic_ids

    def predict(
        self, texts: Union[str, List[str]], word_segment: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Unified prediction interface conforming strictly to PROJECT.md § Interface Contracts.

        Args:
            texts: Single string or list of Vietnamese feedback sentences.
            word_segment: If True, applies idempotent PyVi segmentation.

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

        sent_probs, topic_probs = self.predict_proba(input_list, word_segment=word_segment)

        results: List[Dict[str, Any]] = []
        for i in range(len(input_list)):
            s_row = sent_probs[i]
            t_row = topic_probs[i]

            s_id = int(np.argmax(s_row))
            t_id = int(np.argmax(t_row))

            s_conf = float(s_row[s_id])
            t_conf = float(t_row[t_id])

            s_dist = {SENTIMENT_MAP[c]: float(s_row[c]) for c in range(len(SENTIMENT_MAP))}
            t_dist = {TOPIC_MAP[c]: float(t_row[c]) for c in range(len(TOPIC_MAP))}

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

    def save(self, path: str = "saved_models/baseline.joblib") -> str:
        """
        Serializes the model into a joblib file. Automatically creates directory.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model. Fit before saving.")
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        joblib.dump(self, abs_path)
        return abs_path

    @classmethod
    def load(cls, path: str = "saved_models/baseline.joblib") -> "VietnameseFeedbackBaseline":
        """
        Deserializes model from disk with backwards-compatible state loading.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Baseline model checkpoint not found at '{path}'.")
        loaded = joblib.load(path)
        if isinstance(loaded, cls):
            return loaded
        elif isinstance(loaded, dict):
            instance = cls()
            instance.vectorizer = loaded["vectorizer"]
            instance.sentiment_clf = loaded["sentiment_clf"]
            instance.topic_clf = loaded["topic_clf"]
            instance.is_fitted = loaded.get("is_fitted", True)
            return instance
        else:
            raise TypeError(f"Unrecognized serialized object type: {type(loaded)}")
