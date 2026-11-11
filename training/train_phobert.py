"""
training/train_phobert.py
PhoBERT Multi-Task Model Fine-Tuning Pipeline on UIT-VSFC Dataset.
Trains shared PhoBERT backbone + dual linear classification heads with AdamW,
linear warmup and decay, mixed precision (AMP fp16), and best-model checkpointing.
Conforms strictly to PROJECT.md § Architecture and Interface Contracts.
"""

import argparse
from collections import Counter
import gc
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
import torch.nn.functional as F
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


def compute_smoothed_class_weights(
    labels: List[int],
    num_classes: int,
    power: float = 0.35,
) -> Optional[torch.Tensor]:
    """Computes normalized power-smoothed inverse-frequency class weights for imbalanced tasks."""
    if power <= 0.0 or not labels:
        return None
    counts = Counter(int(x) for x in labels)
    if len(counts) < num_classes:
        return None
    max_count = max(counts.values())
    weights = torch.tensor(
        [(max_count / max(1, counts[c])) ** power for c in range(num_classes)],
        dtype=torch.float32,
    )
    return weights / weights.mean()


def _dynamic_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collates samples while trimming padding dynamically to the longest sequence in the batch."""
    keys = batch[0].keys()
    collated: Dict[str, torch.Tensor] = {k: torch.stack([item[k] for item in batch]) for k in keys}
    if "attention_mask" in collated and "input_ids" in collated:
        max_seq = max(1, int(collated["attention_mask"].sum(dim=1).max().item()))
        collated["input_ids"] = collated["input_ids"][:, :max_seq]
        collated["attention_mask"] = collated["attention_mask"][:, :max_seq]
    return collated


def _build_bucketed_loader(
    dataset: Any,
    batch_size: int,
    shuffle_batches: bool,
    seed: int = 42,
) -> DataLoader:
    """Builds a DataLoader grouped by sequence length to minimize padding overhead on CPU/GPU."""
    lengths = dataset.attention_mask.sum(dim=1).tolist()
    rng = random.Random(seed)
    if shuffle_batches:
        indices = sorted(range(len(lengths)), key=lambda i: lengths[i] + rng.uniform(0.0, 4.0))
        batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
        rng.shuffle(batches)
    else:
        indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    return DataLoader(dataset, batch_sampler=batches, collate_fn=_dynamic_collate_fn)


def _extract_frozen_layer_cache(
    model: PhoBERTMultiTask,
    loader: DataLoader,
    device: torch.device,
    freeze_layers: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Pre-computes deterministic bottom-layer hidden states once for frozen-backbone fine-tuning."""
    model.eval()
    encoder_layers = model.phobert.encoder.layer[:freeze_layers]
    cached: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Caching Frozen Layers [0..{freeze_layers - 1}]", unit="batch", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emb = model.phobert.embeddings(input_ids=input_ids)
            ext_mask, _ = model.phobert._create_attention_masks(
                attention_mask=attention_mask,
                encoder_attention_mask=None,
                embedding_output=emb,
                encoder_hidden_states=None,
                past_key_values=None,
            )
            hidden = emb
            for layer in encoder_layers:
                hidden = layer(hidden, attention_mask=ext_mask)
            cached.append((
                hidden.to(dtype=torch.float16).cpu(),
                ext_mask.cpu(),
                batch["sentiment_labels"].cpu(),
                batch["topic_labels"].cpu(),
            ))
    return cached


def _compute_multitask_loss(
    sentiment_logits: torch.Tensor,
    topic_logits: torch.Tensor,
    sentiment_labels: torch.Tensor,
    topic_labels: torch.Tensor,
    sentiment_weight: Optional[torch.Tensor],
    topic_weight: Optional[torch.Tensor],
    label_smoothing: float,
) -> torch.Tensor:
    """Computes unweighted or smoothed class-weighted multi-task cross-entropy loss."""
    loss_sent = F.cross_entropy(
        sentiment_logits,
        sentiment_labels.long(),
        weight=sentiment_weight,
        label_smoothing=label_smoothing,
    )
    loss_topic = F.cross_entropy(
        topic_logits,
        topic_labels.long(),
        weight=topic_weight,
        label_smoothing=label_smoothing,
    )
    return loss_sent + loss_topic


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
    batch_size: int = 32,
    lr: float = 3e-5,
    head_lr: Optional[float] = None,
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
    freeze_bottom_layers: int = -1,
    class_weight_power: float = 0.35,
    label_smoothing: float = 0.03,
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
    if device.type == "cpu":
        optimal_threads = min(max(1, (os.cpu_count() or 6) // 2), 6)
        torch.set_num_threads(optimal_threads)

    resolved_head_lr = head_lr if head_lr is not None else max(lr * 15.0, 4e-4)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 2. Tokenizer & DataLoaders
    print(f"[1/5] Loading tokenizer '{model_name}' (use_fast=False)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print(f"[2/5] Initializing UIT-VSFC DataLoaders from '{data_dir}'...")
    raw_train_loader, raw_val_loader, raw_test_loader = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        data_dir=data_dir,
        word_segment=True,
        num_workers=0,
        quick_mode=quick_run,
    )

    val_loader = _build_bucketed_loader(raw_val_loader.dataset, batch_size=max(batch_size, 64), shuffle_batches=False, seed=seed)
    test_loader = _build_bucketed_loader(raw_test_loader.dataset, batch_size=max(batch_size, 64), shuffle_batches=False, seed=seed)

    # 3. Model Architecture Initialization
    print(f"[3/5] Instantiating PhoBERTMultiTask model...")
    model = PhoBERTMultiTask(
        model_name_or_path=model_name,
        num_sentiment_classes=3,
        num_topic_classes=4,
        dropout_rate=0.2,
    )
    model.to(device)

    total_encoder_layers = len(getattr(getattr(model.phobert, "encoder", None), "layer", []))
    if freeze_bottom_layers < 0:
        resolved_freeze = 6 if (device.type == "cpu" and not quick_run and total_encoder_layers >= 12) else 0
    else:
        resolved_freeze = min(freeze_bottom_layers, max(0, total_encoder_layers - 1))

    if resolved_freeze > 0:
        for param in model.phobert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.phobert.encoder.layer[:resolved_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

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
        print(f"  Mixed Precision: Disabled (CPU execution, threads={torch.get_num_threads()})")
    print(f"  Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, head_lr={resolved_head_lr}, max_len={max_length}")
    print(f"  Frozen Bottom:   {resolved_freeze}/{total_encoder_layers} encoder layers (Cached Top-Layer Mode: {resolved_freeze > 0 and device.type == 'cpu'})")
    print(f"  Quick Run Mode:  {quick_run}")
    print(f"  Checkpoint Path: {os.path.abspath(save_dir)}")
    print("=" * 65 + "\n")

    # Compute smoothed class weights on training split
    train_sent_list = raw_train_loader.dataset.sentiment_labels.tolist()
    train_topic_list = raw_train_loader.dataset.topic_labels.tolist()
    eff_power = 0.0 if quick_run else class_weight_power
    eff_smoothing = 0.0 if quick_run else label_smoothing
    s_weights = compute_smoothed_class_weights(train_sent_list, num_classes=3, power=eff_power)
    t_weights = compute_smoothed_class_weights(train_topic_list, num_classes=4, power=eff_power)
    if s_weights is not None:
        s_weights = s_weights.to(device)
    if t_weights is not None:
        t_weights = t_weights.to(device)

    use_cached_top = (resolved_freeze > 0) and (device.type == "cpu") and (not quick_run)
    cached_batches: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None
    if use_cached_top:
        bucket_train_loader = _build_bucketed_loader(raw_train_loader.dataset, batch_size=batch_size, shuffle_batches=True, seed=seed)
        t_cache = time.time()
        cached_batches = _extract_frozen_layer_cache(model, bucket_train_loader, device, resolved_freeze)
        gc.collect()
        num_train_batches = len(cached_batches)
        print(f"      Cached {num_train_batches:,} bucketed batches through layers [0..{resolved_freeze - 1}] in {time.time() - t_cache:.1f}s")
    else:
        num_train_batches = len(raw_train_loader)
        print(f"      Train batches: {num_train_batches:,} | Val batches: {len(val_loader):,} | Test batches: {len(test_loader):,}")

    # 4. Differential Optimizer & Linear Warmup Scheduler
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    head_prefixes = ("sentiment_classifier.", "topic_classifier.")
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head_prefixes) and not any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head_prefixes) and any(nd in n for nd in no_decay)],
            "lr": lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(head_prefixes) and not any(nd in n for nd in no_decay)],
            "lr": resolved_head_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(head_prefixes) and any(nd in n for nd in no_decay)],
            "lr": resolved_head_lr,
            "weight_decay": 0.0,
        },
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    total_training_steps = num_train_batches * epochs
    num_warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"      Total training steps: {total_training_steps:,} (Warmup steps: {num_warmup_steps:,})")

    # 5. Training and Validation Loop
    print(f"\n[4/5] Commencing Fine-Tuning across {epochs} Epoch(s)...")
    best_mean_macro_f1 = -1.0
    best_epoch = 0
    best_val_metrics: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []
    rng = random.Random(seed)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        if resolved_freeze > 0:
            model.phobert.embeddings.eval()
            for layer in model.phobert.encoder.layer[:resolved_freeze]:
                layer.eval()

        running_train_loss = 0.0
        train_samples = 0

        if cached_batches is not None:
            rng.shuffle(cached_batches)
            pbar = tqdm(cached_batches, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", unit="batch", leave=False)
            top_layers = model.phobert.encoder.layer[resolved_freeze:]
            for hidden_half, ext_mask_cpu, s_cpu, t_cpu in pbar:
                optimizer.zero_grad()
                hidden = hidden_half.to(device=device, dtype=torch.float32)
                ext_mask = ext_mask_cpu.to(device=device)
                s_labels = s_cpu.to(device=device)
                t_labels = t_cpu.to(device=device)

                for layer in top_layers:
                    hidden = layer(hidden, attention_mask=ext_mask)
                cls_rep = model.dropout(hidden[:, 0, :])
                s_logits = model.sentiment_classifier(cls_rep)
                t_logits = model.topic_classifier(cls_rep)
                loss = _compute_multitask_loss(
                    s_logits, t_logits, s_labels, t_labels, s_weights, t_weights, eff_smoothing
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()

                batch_size_curr = s_labels.size(0)
                running_train_loss += loss.item() * batch_size_curr
                train_samples += batch_size_curr
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
        else:
            epoch_train_loader = (
                raw_train_loader
                if quick_run
                else _build_bucketed_loader(raw_train_loader.dataset, batch_size=batch_size, shuffle_batches=True, seed=seed + epoch)
            )
            pbar = tqdm(epoch_train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]", unit="batch", leave=False)
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                s_labels = batch["sentiment_labels"].to(device)
                t_labels = batch["topic_labels"].to(device)

                optimizer.zero_grad()
                try:
                    autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp)
                except (AttributeError, TypeError):
                    autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)

                with autocast_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = _compute_multitask_loss(
                        outputs["sentiment_logits"],
                        outputs["topic_logits"],
                        s_labels,
                        t_labels,
                        s_weights,
                        t_weights,
                        eff_smoothing,
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                batch_size_curr = input_ids.size(0)
                running_train_loss += loss.item() * batch_size_curr
                train_samples += batch_size_curr
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

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
                "head_lr": resolved_head_lr,
                "freeze_bottom_layers": resolved_freeze,
                "class_weight_power": eff_power,
                "label_smoothing": eff_smoothing,
                "quick_run": quick_run,
                "saved_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2)

    if cached_batches is not None:
        del cached_batches
        gc.collect()

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
        default=32,
        help="Batch size for training and evaluation (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for AdamW encoder layers (default: 3e-5)",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=None,
        help="Optional higher learning rate for dual classification heads (default: max(lr*15, 4e-4))",
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
        "--freeze_bottom_layers",
        type=int,
        default=-1,
        help="Number of bottom transformer layers to freeze (-1 = auto: 6 on CPU, 0 on CUDA)",
    )
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=0.35,
        help="Power exponent for smoothed inverse class weights (default: 0.35)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.03,
        help="Cross-entropy label smoothing factor (default: 0.03)",
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
        head_lr=args.head_lr,
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
        freeze_bottom_layers=args.freeze_bottom_layers,
        class_weight_power=args.class_weight_power,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()
