# Project: Text Classification of Vietnamese Student Feedback

## Architecture

The project delivers a clean, student-friendly machine learning system comparing a Classical Baseline (TF-IDF + Dual Logistic Regression) against a fine-tuned Transformer (PhoBERT v2 Multi-Task) on the UIT-VSFC dataset for Sentiment and Topic classification, accompanied by an interactive Streamlit demo dashboard.

```
vietnamese-feedback-classifier/
├── data/
│   ├── download_data.py          # Auto-acquires UIT-VSFC parquet splits into local cache
│   └── dataset.py                # PyVi segmentation, Torch Dataset, DataLoaders
├── models/
│   ├── baseline_model.py         # TF-IDF + Dual LogisticRegression heads
│   └── phobert_model.py          # PhoBERTMultiTask with shared CLS encoder & dual linear heads
├── training/
│   ├── train_baseline.py         # Trains classical baseline in ~2s, saves to saved_models/baseline.joblib
│   └── train_phobert.py          # Fine-tunes PhoBERT (AMP fp16, GPU/CPU auto-detect, AdamW)
├── evaluation/
│   └── evaluate.py               # Evaluates test split, outputs Accuracy & Macro-F1 tables & JSON
├── app/
│   └── app.py                    # Streamlit interactive feedback analyzer with side-by-side mode
├── tests/                        # Comprehensive E2E opaque-box test suite (Tiers 1-4)
│   ├── test_runner.py            # Automated test suite runner
│   ├── test_tier1_features.py    # Tier 1: Per-feature equivalence tests
│   ├── test_tier2_boundaries.py  # Tier 2: Boundary & corner cases
│   ├── test_tier3_pairwise.py    # Tier 3: Cross-feature combinations
│   └── test_tier4_workload.py    # Tier 4: Real-world student feedback scenarios
├── saved_models/                 # Serialized weights (baseline.joblib, phobert_multitask/)
├── results/                      # Evaluation reports (comparison_summary.md, metrics.json)
├── requirements.txt              # Pinned student-friendly dependencies
└── README.md                     # Minimal, step-by-step CLI usage guide
```

### Data & Model Flow
1. **Raw Text** -> **PyVi Word Segmentation** (`ViTokenizer.tokenize`).
2. **Classical Track**: Segmented text -> Shared `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)` -> Dual `LogisticRegression(class_weight='balanced')` -> Sentiment (3 classes) & Topic (4 classes) probabilities.
3. **PhoBERT Track**: Segmented text -> `vinai/phobert-base-v2` Tokenizer -> Shared 768-d CLS representation -> Sentiment Linear Head (768->3) & Topic Linear Head (768->4) -> Cross-Entropy Multi-Task Loss.
4. **Evaluation**: Both models evaluated on the official test set ($N=3,166$) reporting Accuracy and Macro-F1.
5. **Dashboard**: Streamlit web app providing single-model and side-by-side analysis, color-coded badges, confidence bars, latency timers, and 8 pre-loaded feedback presets.

---

## Feature Inventory

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | UIT-VSFC Schema & Columns | `sentence` (string), `sentiment` (0,1,2), `topic` (0,1,2,3) with zero missing values | M1 | Survey (Explorer 1) |
| F2 | Academic Splits Standard | Fixed train (11,426), val (1,583), test (3,166) splits without cross-split leakage | M1 | Survey (Explorer 1) |
| F3 | 3-Class Sentiment Task | Negative (0), Neutral (1), Positive (2) mapping and metrics | M6 (E2E) | Survey (Explorer 1, 3) |
| F4 | 4-Class Topic Task | Lecturer (0), Training Program (1), Facility (2), Others (3) mapping and metrics | M6 (E2E) | Survey (Explorer 1, 3) |
| F5 | Automated Data Acquisition | Fast download from Hugging Face Parquet endpoint with local caching in `data/uit_vsfc/` | M1 | Survey (Explorer 1) |
| F6 | Vietnamese Word Segmentation | Pure-Python word segmentation via `pyvi.ViTokenizer` joining compound words with `_` | M1 | Survey (Explorer 1, 2) |
| F7 | Classical Baseline Pipeline | TF-IDF (1-2 grams, sublinear_tf) + dual balanced LogisticRegression classifiers | M2 | Survey (Explorer 2) |
| F8 | Baseline Serialization | Fast persistence and deserialization via `saved_models/baseline.joblib` (<8MB) | M2 | Survey (Explorer 2) |
| F9 | PhoBERT Multi-Task Architecture | Shared `vinai/phobert-base-v2` encoder with dual linear heads on pooled `[CLS]` token | M3 | Survey (Explorer 2) |
| F10 | PhoBERT Fine-Tuning Pipeline | Multi-task loss training with AdamW, linear schedule, AMP fp16, GPU/CPU auto-detect | M3 | Survey (Explorer 2) |
| F11 | Unified Prediction Interface | Standardized output schema returning labels, confidence scores, and probability distributions | M2, M3 | Survey (Explorer 2, 3) |
| F12 | Test Split Evaluation Script | Standalone `evaluation/evaluate.py` calculating Accuracy and Macro-F1 on test set | M4 | Survey (Explorer 3) |
| F13 | Comparative Reporting Artifacts | Generates aligned CLI table, `results/comparison_summary.md`, and `results/metrics.json` | M4 | Survey (Explorer 3) |
| F14 | Streamlit Interactive Dashboard | Clean web UI with text input box, live prediction triggers, and responsive layout | M5 | Survey (Explorer 3) |
| F15 | Model Comparison Modes | Radio button toggle for Classical Baseline, PhoBERT, or Side-by-Side simultaneous comparison | M5 | Survey (Explorer 3) |
| F16 | Visual Cards & Probability Bars | Color-coded sentiment badges, topic tags, emoji indicators, and interactive progress bars | M5 | Survey (Explorer 3) |
| F17 | Curated Feedback Presets | 8 authentic Vietnamese student feedback examples pre-loaded into sidebar/dropdown | M5 | Survey (Explorer 3) |
| F18 | Student Ergonomics & Scope Discipline | Clean, flat code layout with standard CLI commands; zero prohibited 15-20p reports or slides | M4, M5 | Survey (Explorer 3) |
| F19 | Multi-Tier E2E Test Suite | Comprehensive opaque-box test suite (Tiers 1-4) & Tier 5 adversarial verification | E2E Track & M6 | Survey & Project Pattern |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| **M1** | Data Acquisition & Preprocessing Pipeline | `data/download_data.py`, `data/dataset.py`, `requirements.txt` | None | DONE |
| **M2** | Classical Baseline Model & Training | `models/baseline_model.py`, `training/train_baseline.py`, `saved_models/baseline.joblib` | M1 | DONE |
| **M3** | PhoBERT Multi-Task Model & Fine-Tuning | `models/phobert_model.py`, `training/train_phobert.py`, `tests/test_phobert.py`, `saved_models/phobert_multitask/` | M1 | DONE |
| **M4** | Comparative Evaluation & Metrics Script | `evaluation/evaluate.py`, `results/comparison_summary.md`, `results/metrics.json` | M2, M3 | DONE |
| **M5** | Interactive Streamlit Demo Dashboard | `app/app.py`, preset examples library, side-by-side mode | M2, M3 | DONE |
| **M6** | Final Acceptance & Student Documentation | Clean `README.md`, full E2E verification, R5 scope audit | M1-M5 | DONE |

### Parallel Track: E2E Testing Suite
- **E2E Track**: Design and build independent opaque-box test harness (`tests/test_runner.py`, `tests/test_tier*.py`) validating requirements R1-R5, producing `TEST_INFRA.md` and `TEST_READY.md`. Runs concurrently with Implementation Track.

---

## Interface Contracts

### 1. Data Pipeline (`data/dataset.py`)
```python
def load_vsfc_split(split_name: str, cache_dir: str = "data/uit_vsfc") -> Tuple[List[str], List[int], List[int]]:
    """Loads a split ('train', 'validation', 'test').
    Returns:
        texts: List[str] (pre-segmented with pyvi)
        sentiments: List[int] in {0: negative, 1: neutral, 2: positive}
        topics: List[int] in {0: lecturer, 1: training_program, 2: facility, 3: others}
    """
```

### 2. Unified Prediction Output Schema
Both `VietnameseFeedbackBaseline.predict(texts)` and `PhoBERTMultiTask` predictor MUST produce:
```python
{
    "sentiment": {
        "label": str,       # "negative" | "neutral" | "positive"
        "label_id": int,    # 0 | 1 | 2
        "confidence": float,# 0.0 - 1.0
        "distribution": {   # Dict[str, float] summing to 1.0
            "negative": float,
            "neutral": float,
            "positive": float
        }
    },
    "topic": {
        "label": str,       # "lecturer" | "training_program" | "facility" | "others"
        "label_id": int,    # 0 | 1 | 2 | 3
        "confidence": float,# 0.0 - 1.0
        "distribution": {   # Dict[str, float] summing to 1.0
            "lecturer": float,
            "training_program": float,
            "facility": float,
            "others": float
        }
    }
}
```

### 3. Evaluation Output Contract (`evaluation/evaluate.py`)
- CLI execution: `python evaluation/evaluate.py [--split test] [--save_reports]`
- Outputs:
  - Aligned terminal table reporting `Sentiment Accuracy`, `Sentiment Macro-F1`, `Topic Accuracy`, `Topic Macro-F1`.
  - `results/comparison_summary.md` (Markdown format with Delta comparisons).
  - `results/metrics.json` (Structured scores for automated grading).

### 4. Interactive Dashboard Contract (`app/app.py`)
- CLI command: `python -m streamlit run app/app.py`
- Model caching: `@st.cache_resource` for weights & tokenizers.
- Graceful fallback: If PhoBERT weights are not present, fallback cleanly with UI warning to Baseline without application crash.

---

## Code Layout & Write Ownership

| Path | Exclusive Owner Milestone | Description |
|---|---|---|
| `requirements.txt` | M1 | Pinned student-friendly dependencies |
| `data/download_data.py` | M1 | Dataset downloader from Hugging Face Parquet endpoint |
| `data/dataset.py` | M1 | Dataset loader & PyVi segmentation |
| `models/baseline_model.py` | M2 | Classical baseline model class |
| `training/train_baseline.py` | M2 | Baseline training script |
| `models/phobert_model.py` | M3 | PhoBERT multi-task PyTorch module |
| `training/train_phobert.py` | M3 | PhoBERT fine-tuning script |
| `evaluation/evaluate.py` | M4 | Evaluation and comparative reporting script |
| `app/app.py` | M5 | Streamlit interactive feedback analyzer dashboard |
| `tests/` | E2E Testing Track | Independent opaque-box test suite (Tiers 1-4) |
| `README.md` | M5 / M6 | Step-by-step student CLI instructions |
