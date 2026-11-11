"""
training/train_phobert.py
PhoBERT Multi-Task Model Fine-Tuning Pipeline on UIT-VSFC Dataset.
Trains shared PhoBERT backbone + dual linear classification heads with AdamW,
linear warmup and decay, mixed precision (AMP fp16), and best-model checkpointing.
Conforms strictly to PROJECT.md § Architecture and Interface Contracts.
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Suppress Hugging Face Windows symlink warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Windows console encoding safeguard for Vietnamese characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path for direct CLI invocation
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from data.dataset import SENTIMENT_MAP, TOPIC_MAP, get_dataloaders
from models.phobert_model import PhoBERTMultiTask


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: PhoBERTMultiTask,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "validation",
    show_report: bool = False,
) -> Dict[str, float]:
    """
    Evaluates PhoBERTMultiTask on a given DataLoader, calculating Accuracy and Macro-F1
    for both Sentiment and Topic classification tasks.

    Args:
        model: PhoBERTMultiTask instance.
        dataloader: PyTorch DataLoader yielding VSFC batches.
        device: Device to run evaluation on.
        split_name: Name of the evaluated split ('validation' or 'test').
        show_report: If True, prints detailed per-class classification reports.

    Returns:
        Dictionary of computed metrics (loss, accuracies, macro-F1s, mean macro-F1).
    """
    model.eval()
    total_loss = 0.0
    all_s_true: List[int] = []
    all_s_pred: List[int] = []
    all_t_true: List[int] = []
    all_t_pred: List[int] = []

    t0 = time.time()
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating [{split_name.upper()}]", unit="batch", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            s_labels = batch.get("sentiment_labels")
            t_labels = batch.get("topic_labels")

            if s_labels is not None:
                s_labels = s_labels.to(device)
            if t_labels is not None:
                t_labels = t_labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_labels=s_labels,
                topic_labels=t_labels,
            )

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item() * input_ids.size(0)

            s_logits = outputs["sentiment_logits"]
            t_logits = outputs["topic_logits"]

            s_preds = torch.argmax(s_logits, dim=-1).cpu().tolist()
            t_preds = torch.argmax(t_logits, dim=-1).cpu().tolist()

            all_s_pred.extend(s_preds)
            all_t_pred.extend(t_preds)

            if s_labels is not None:
                all_s_true.extend(s_labels.cpu().tolist())
            if t_labels is not None:
                all_t_true.extend(t_labels.cpu().tolist())

    eval_time = time.time() - t0
    num_samples = len(all_s_pred)
    avg_loss = (total_loss / num_samples) if num_samples > 0 else 0.0
    throughput = (num_samples / eval_time) if eval_time > 0 else 0.0

    # Calculate metrics
    s_acc = float(accuracy_score(all_s_true, all_s_pred)) if all_s_true else 0.0
    s_f1 = float(f1_score(all_s_true, all_s_pred, average="macro", zero_division=0)) if all_s_true else 0.0
    t_acc = float(accuracy_score(all_t_true, all_t_pred)) if all_t_true else 0.0
    t_f1 = float(f1_score(all_t_true, all_t_pred, average="macro", zero_division=0)) if all_t_true else 0.0
    mean_macro_f1 = float((s_f1 + t_f1) / 2.0)

    # Print formatted metrics table matching project standard
    print(f"\n[{split_name.upper()} EVALUATION METRICS] ({num_samples:,} samples, {eval_time:.2f}s, {throughput:.1f} samples/sec)")
    print("-" * 65)
    print(f"  {'Task':<24} {'Accuracy':<14} {'Macro-F1':<14}")
    print("-" * 65)
    print(f"  {'Sentiment (3 classes)':<24} {s_acc * 100:6.2f}%        {s_f1 * 100:6.2f}%")
    print(f"  {'Topic (4 classes)':<24} {t_acc * 100:6.2f}%        {t_f1 * 100:6.2f}%")
    print("-" * 65)
    print(f"  Combined Mean Macro-F1:  {mean_macro_f1 * 100:6.2f}% | Average Loss: {avg_loss:.4f}")
    print("-" * 65)

    if show_report and all_s_true and all_t_true:
        sent_names = [SENTIMENT_MAP[i] for i in range(len(SENTIMENT_MAP))]
        topic_names = [TOPIC_MAP[i] for i in range(len(TOPIC_MAP))]
        print(f"\nDetailed Classification Report — Sentiment ({split_name}):")
        print(classification_report(all_s_true, all_s_pred, target_names=sent_names, digits=4, zero_division=0))
        print(f"Detailed Classification Report — Topic ({split_name}):")
        print(classification_report(all_t_true, all_t_pred, target_names=topic_names, digits=4, zero_division=0))

    return {
        "loss": avg_loss,
        "sentiment_accuracy": s_acc,
        "sentiment_macro_f1": s_f1,
        "topic_accuracy": t_acc,
        "topic_macro_f1": t_f1,
        "mean_macro_f1": mean_macro_f1,
        "eval_time": eval_time,
        "num_samples": num_samples,
    }


def train_phobert(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    model_name: str = "vinai/phobert-base-v2",
    save_dir: str = "saved_models/phobert_multitask",
    data_dir: str = "data/uit_vsfc",
    quick_run: bool = False,
    no_amp: bool = False,
    eval_test: bool = True,
    show_report: bool = False,
    seed: int = 42,
) -> Tuple[PhoBERTMultiTask, Dict[str, Any]]:
    """
    Main training function for fine-tuning PhoBERT Multi-Task model on UIT-VSFC.

    Returns:
        Tuple of (best_model, training_summary_dict).
    """
    set_seed(seed)
    total_start_time = time.time()

    # 1. Device and Mixed Precision Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not no_amp)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("=" * 65)
    print("  PHOBERT MULTI-TASK FINE-TUNING PIPELINE")
    print(f"  Backbone: {model_name} | Shared CLS + Dual Heads")
    print("=" * 65)
    print(f"  Device:          {device.type.upper()}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU Name:        {gpu_name} ({vram_gb:.2f} GB VRAM)")
        print(f"  Mixed Precision: {'AMP fp16 Enabled' if use_amp else 'Disabled'}")
    else:
        print("  Mixed Precision: Disabled (CPU execution)")
        if not quick_run:
            print("  Note: CPU training on full dataset is slow. Consider --quick_run for tests.")
    print(f"  Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, max_len={max_length}")
    print(f"  Quick Run Mode:  {quick_run}")
    print(f"  Checkpoint Path: {os.path.abspath(save_dir)}")
    print("=" * 65 + "\n")

    # 2. Tokenizer & DataLoaders
    print(f"[1/5] Loading tokenizer '{model_name}' (use_fast=False)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print(f"[2/5] Initializing UIT-VSFC DataLoaders from '{data_dir}'...")
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        data_dir=data_dir,
        word_segment=True,
        num_workers=0,
        quick_mode=quick_run,
    )
    print(f"      Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,} | Test batches: {len(test_loader):,}")

    # 3. Model Architecture Initialization
    print(f"[3/5] Instantiating PhoBERTMultiTask model...")
    model = PhoBERTMultiTask(
        model_name_or_path=model_name,
        num_sentiment_classes=3,
        num_topic_classes=4,
        dropout_rate=0.2,
    )
    model.to(device)

    # 4. Optimizer & Linear Warmup Scheduler
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    total_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )
    print(f"      Total training steps: {total_training_steps:,} (Warmup steps: {num_warmup_steps:,})")

    # 5. Training and Validation Loop
    print(f"\n[4/5] Commencing Fine-Tuning across {epochs} Epoch(s)...")
    best_mean_macro_f1 = -1.0
    best_epoch = 0
    best_val_metrics: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]",
            unit="batch",
            leave=False,
        )

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            s_labels = batch["sentiment_labels"].to(device)
            t_labels = batch["topic_labels"].to(device)

            optimizer.zero_grad()

            # Autocast context for mixed precision
            try:
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentiment_labels=s_labels,
                    topic_labels=t_labels,
                )
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            batch_size_curr = input_ids.size(0)
            running_train_loss += loss.item() * batch_size_curr
            train_samples += batch_size_curr

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = (running_train_loss / train_samples) if train_samples > 0 else 0.0
        print(f"\n>>> Epoch {epoch:02d}/{epochs:02d} Finished in {epoch_duration:.2f}s | Average Train Loss: {avg_train_loss:.4f}")

        # Run validation evaluation
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            split_name="validation",
            show_report=show_report,
        )

        current_val_mean_f1 = val_metrics["mean_macro_f1"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_sentiment_acc": val_metrics["sentiment_accuracy"],
            "val_sentiment_f1": val_metrics["sentiment_macro_f1"],
            "val_topic_acc": val_metrics["topic_accuracy"],
            "val_topic_f1": val_metrics["topic_macro_f1"],
            "val_mean_macro_f1": current_val_mean_f1,
            "epoch_duration": epoch_duration,
        }
        history.append(epoch_record)

        # Checkpointing: Save if best validation Mean Macro-F1 is achieved
        if current_val_mean_f1 > best_mean_macro_f1:
            best_mean_macro_f1 = current_val_mean_f1
            best_epoch = epoch
            best_val_metrics = val_metrics

            print(f"  [*] New Best Validation Mean Macro-F1: {best_mean_macro_f1 * 100:.2f}% (Epoch {epoch})")
            print(f"      Saving checkpoint & tokenizer to '{save_dir}'...")

            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            meta_data = {
                "best_epoch": best_epoch,
                "best_val_mean_macro_f1": best_mean_macro_f1,
                "val_metrics": val_metrics,
                "model_name": model_name,
                "max_length": max_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "quick_run": quick_run,
                "saved_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

    # 6. Post-training Evaluation on Test Split
    test_metrics: Dict[str, float] = {}
    if eval_test and os.path.isdir(save_dir):
        print(f"\n[5/5] Loading Best Checkpoint (Epoch {best_epoch}) for Test Split Evaluation...")
        best_model = PhoBERTMultiTask.from_pretrained(save_dir, device=device)
        test_metrics = evaluate(
            model=best_model,
            dataloader=test_loader,
            device=device,
            split_name="test",
            show_report=show_report,
        )

        with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 65)
    print(f"  PHOBERT FINE-TUNING COMPLETE ({total_duration:.2f}s total)")
    print(f"  Best Validation Epoch:     {best_epoch}")
    print(f"  Best Validation Mean F1:   {best_mean_macro_f1 * 100:.2f}%")
    if test_metrics:
        print(f"  Test Sentiment Macro-F1:   {test_metrics.get('sentiment_macro_f1', 0.0) * 100:.2f}%")
        print(f"  Test Topic Macro-F1:       {test_metrics.get('topic_macro_f1', 0.0) * 100:.2f}%")
    print(f"  Checkpoint Directory:      {os.path.abspath(save_dir)}")
    print("=" * 65 + "\n")

    summary = {
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "total_duration": total_duration,
        "save_dir": os.path.abspath(save_dir),
    }

    # Save complete training history
    with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return model, summary


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune PhoBERT Multi-Task Model for Sentiment and Topic Classification on UIT-VSFC"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimizer (default: 2e-5)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization (default: 128)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Ratio of total training steps for linear warmup (default: 0.1)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vinai/phobert-base-v2",
        help="Pretrained PhoBERT model identifier (default: vinai/phobert-base-v2)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models/phobert_multitask",
        help="Destination directory for best checkpoint and tokenizer (default: saved_models/phobert_multitask)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/uit_vsfc",
        help="Path to UIT-VSFC dataset splits directory (default: data/uit_vsfc)",
    )
    parser.add_argument(
        "--quick_run",
        action="store_true",
        help="Quick run mode with small subset of batches for rapid pipeline verification",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable AMP fp16 mixed precision even if CUDA is available",
    )
    parser.add_argument(
        "--no_test",
        action="store_true",
        help="Skip final evaluation on test split after training",
    )
    parser.add_argument(
        "--show_report",
        action="store_true",
        help="Display detailed classification report with per-class metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    train_phobert(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        model_name=args.model_name,
        save_dir=args.save_dir,
        data_dir=args.data_dir,
        quick_run=args.quick_run,
        no_amp=args.no_amp,
        eval_test=not args.no_test,
        show_report=args.show_report,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
