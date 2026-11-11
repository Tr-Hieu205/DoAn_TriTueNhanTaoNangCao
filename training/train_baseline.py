"""
training/train_baseline.py
Classical Baseline Model Training and Evaluation Script on UIT-VSFC Dataset.
Trains TF-IDF + Dual Balanced Logistic Regression in ~2 seconds and serializes
the trained artifact to saved_models/baseline.joblib.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Windows console encoding safeguard for Vietnamese characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path for direct CLI execution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import accuracy_score, classification_report, f1_score
from data.dataset import SENTIMENT_MAP, TOPIC_MAP, load_vsfc_split
from models.baseline_model import VietnameseFeedbackBaseline


def evaluate_split(
    model: VietnameseFeedbackBaseline,
    texts: List[str],
    sentiments: List[int],
    topics: List[int],
    split_name: str = "validation",
    show_report: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates model on a given split and computes Accuracy and Macro-F1.

    Args:
        model: Fitted VietnameseFeedbackBaseline instance.
        texts: List of feedback sentences.
        sentiments: Ground truth sentiment integer labels (0, 1, 2).
        topics: Ground truth topic integer labels (0, 1, 2, 3).
        split_name: Name of the split ('validation' or 'test').
        show_report: If True, prints detailed per-class classification report.

    Returns:
        Dictionary containing computed metrics and inference timing.
    """
    t0 = time.time()
    sent_preds, topic_preds = model.predict_labels(texts, word_segment=False)
    infer_time = time.time() - t0

    s_acc = float(accuracy_score(sentiments, sent_preds))
    s_f1 = float(f1_score(sentiments, sent_preds, average="macro", zero_division=0))
    t_acc = float(accuracy_score(topics, topic_preds))
    t_f1 = float(f1_score(topics, topic_preds, average="macro", zero_division=0))

    throughput = len(texts) / infer_time if infer_time > 0 else 0.0

    print(f"\n[{split_name.upper()} EVALUATION METRICS] ({len(texts):,} samples)")
    print("-" * 65)
    print(f"  {'Task':<22} {'Accuracy':<14} {'Macro-F1':<14}")
    print("-" * 65)
    print(f"  {'Sentiment (3 classes)':<22} {s_acc * 100:6.2f}%        {s_f1 * 100:6.2f}%")
    print(f"  {'Topic (4 classes)':<22} {t_acc * 100:6.2f}%        {t_f1 * 100:6.2f}%")
    print("-" * 65)
    print(f"  Inference Latency: {infer_time:.3f}s ({throughput:,.1f} samples/sec)")

    if show_report:
        sent_names = [SENTIMENT_MAP[i] for i in range(len(SENTIMENT_MAP))]
        topic_names = [TOPIC_MAP[i] for i in range(len(TOPIC_MAP))]
        print(f"\nClassification Report — Sentiment ({split_name}):")
        print(classification_report(sentiments, sent_preds, target_names=sent_names, digits=4))
        print(f"Classification Report — Topic ({split_name}):")
        print(classification_report(topics, topic_preds, target_names=topic_names, digits=4))

    return {
        "sentiment_accuracy": s_acc,
        "sentiment_macro_f1": s_f1,
        "topic_accuracy": t_acc,
        "topic_macro_f1": t_f1,
        "inference_time": infer_time,
    }


def train_baseline(
    data_dir: str = "data/uit_vsfc",
    save_path: str = "saved_models/baseline.joblib",
    max_features: int = 10000,
    C: float = 1.0,
    max_iter: int = 1000,
    eval_val: bool = True,
    eval_test: bool = True,
    show_report: bool = False,
) -> VietnameseFeedbackBaseline:
    """
    Executes the classical baseline training pipeline:
    1. Loads UIT-VSFC training split with PyVi segmentation.
    2. Fits shared TF-IDF vectorizer + dual balanced LogisticRegression heads.
    3. Evaluates validation split for verification.
    4. Evaluates test split for benchmark metrics.
    5. Saves serialized model artifact to disk.

    Returns:
        The trained VietnameseFeedbackBaseline model instance.
    """
    print("=" * 65)
    print("  UIT-VSFC CLASSICAL BASELINE TRAINING PIPELINE")
    print("  Architecture: TF-IDF (1-2 N-Grams) + Dual Logistic Regression")
    print("=" * 65)

    # 1. Directory creation safeguard
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)

    # 2. Load training split
    print(f"\n[1/4] Loading training data from '{data_dir}' (word_segment=True)...")
    t0_load = time.time()
    train_texts, train_sents, train_topics = load_vsfc_split("train", data_dir=data_dir, word_segment=True)
    load_time = time.time() - t0_load
    print(f"      Loaded {len(train_texts):,} training samples in {load_time:.2f}s")

    # 3. Fit model
    print(f"\n[2/4] Initializing and training VietnameseFeedbackBaseline...")
    print(f"      Hyperparameters: max_features={max_features}, C={C}, max_iter={max_iter}, class_weight='balanced'")
    t0_train = time.time()
    model = VietnameseFeedbackBaseline(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=max_features,
        sublinear_tf=True,
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=42,
    )
    # Texts are already pre-segmented by load_vsfc_split; set word_segment=False to avoid duplicate overhead
    model.fit(train_texts, train_sents, train_topics, word_segment=False)
    train_duration = time.time() - t0_train
    vocab_size = len(model.vectorizer.vocabulary_) if hasattr(model.vectorizer, "vocabulary_") else max_features
    print(f"      Training complete in {train_duration:.2f}s (Vocabulary size: {vocab_size:,} features)")

    # 4. Evaluate validation split
    if eval_val:
        print(f"\n[3/4] Evaluating on Validation split...")
        t0_val = time.time()
        val_texts, val_sents, val_topics = load_vsfc_split("validation", data_dir=data_dir, word_segment=True)
        val_load_time = time.time() - t0_val
        print(f"      Loaded {len(val_texts):,} validation samples in {val_load_time:.2f}s")
        evaluate_split(model, val_texts, val_sents, val_topics, split_name="validation", show_report=show_report)

    # 5. Evaluate test split
    if eval_test:
        print(f"\n[Optional/Test] Evaluating on Test split...")
        t0_test = time.time()
        test_texts, test_sents, test_topics = load_vsfc_split("test", data_dir=data_dir, word_segment=True)
        test_load_time = time.time() - t0_test
        print(f"      Loaded {len(test_texts):,} test samples in {test_load_time:.2f}s")
        evaluate_split(model, test_texts, test_sents, test_topics, split_name="test", show_report=show_report)

    # 6. Save model artifact
    print(f"\n[4/4] Serializing model artifact to '{save_path}'...")
    t0_save = time.time()
    saved_path = model.save(save_path)
    save_duration = time.time() - t0_save
    file_size_bytes = os.path.getsize(saved_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"      Artifact saved successfully: {os.path.abspath(saved_path)}")
    print(f"      Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes) (Serialization time: {save_duration:.2f}s)")

    total_duration = time.time() - t0_load
    print("\n" + "=" * 65)
    print(f"  BASELINE TRAINING PIPELINE COMPLETE ({total_duration:.2f}s total)")
    print("=" * 65 + "\n")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Vietnamese Feedback Classical Baseline Model (TF-IDF + Dual Logistic Regression)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/uit_vsfc",
        help="Path to UIT-VSFC dataset directory (default: data/uit_vsfc)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_models/baseline.joblib",
        help="Destination path for serialized baseline artifact (default: saved_models/baseline.joblib)",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=10000,
        help="Maximum vocabulary features for TF-IDF (default: 10000)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for Logistic Regression (default: 1.0)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum solver iterations (default: 1000)",
    )
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Skip evaluation on the validation split",
    )
    parser.add_argument(
        "--no_test",
        action="store_true",
        help="Skip evaluation on the test split",
    )
    parser.add_argument(
        "--show_report",
        action="store_true",
        help="Display detailed classification report with per-class metrics",
    )

    args = parser.parse_args()

    train_baseline(
        data_dir=args.data_dir,
        save_path=args.save_path,
        max_features=args.max_features,
        C=args.C,
        max_iter=args.max_iter,
        eval_val=not args.no_val,
        eval_test=not args.no_test,
        show_report=args.show_report,
    )


if __name__ == "__main__":
    main()
