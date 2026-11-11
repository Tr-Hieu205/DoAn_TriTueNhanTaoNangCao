# Comparative Evaluation Report: Vietnamese Student Feedback Classification

*Dataset: UIT-VSFC `test` split (N = 3,166 samples)*  
*Generated: 2026-11-11 15:20:05 UTC | Host Device: CPU*

---

## 1. Executive Summary & Benchmark Comparison

| Task / Metric | Classical Baseline (TF-IDF + LR) | PhoBERT Multi-Task (v2) | Delta (Δ) | Relative Improvement (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Sentiment Accuracy** | 86.73% | 93.49% | +6.76% | +7.79% |
| **Sentiment Macro-F1** | 71.90% | 82.89% | +10.99% | +15.29% |
| **Topic Accuracy** | 80.99% | 88.41% | +7.42% | +9.17% |
| **Topic Macro-F1** | 72.72% | 80.10% | +7.39% | +10.16% |
| **Mean Macro-F1** | 72.31% | 81.50% | +9.19% | +12.71% |
| **Inference Latency (ms/sample)** | 0.05 ms | 30.10 ms | +30.05 ms | 660.9x slower |
| **Model Storage Footprint (MB)** | 0.95 MB | 515.06 MB | +514.11 MB | 541.2x larger |

---

## 2. Key Pedagogical Takeaways & Student Insights

### 2.1 Contextual Nuance & Linguistic Representations
- **Bag-of-Words vs Deep Context**:
  The Classical Baseline uses discrete n-gram token counts (`ngram_range=(1,2)`). While highly effective for straightforward keyword feedback ("thầy dạy hay", "phòng học nóng"), it struggles on complex Vietnamese linguistic structures:
  - *Negation and scope inversion*: "Thầy dạy không phải là không hay" contains double negation that confuses linear bag-of-words classifiers.
  - *Compound word ambiguity*: PyVi word segmentation joins compound tokens (`giảng_viên`, `cơ_sở_vật_chất`), but word order and long-distance syntactic dependencies are lost in TF-IDF.
  - *Sarcasm and polite dissatisfaction*: Feedback such as "Cảm ơn nhà trường đã cho chúng em trải nghiệm phòng học như phòng xông hơi" requires deep contextual self-attention across the full sentence to correctly assign Negative sentiment and Facility topic.
- **PhoBERT Multi-Task Architecture**:
  Pre-trained on 20GB of Vietnamese text using RoBERTa architecture, PhoBERT captures bidirectional contextual embeddings. Sharing a single 768-dimensional `[CLS]` representation between sentiment and topic heads enables joint semantic regularization.

### 2.2 Class Imbalance Dynamics & The Macro-F1 Metric (Resolving the Accuracy Paradox)
- **The UIT-VSFC Imbalance Profile**:
  - *Sentiment*: Positive (50.22%) and Negative (44.50%) dominate the dataset; Neutral comprises only **5.27%** (167 / 3,166 samples).
  - *Topic*: Lecturer feedback dominates with **72.33%** (2,290 / 3,166 samples); Training Program accounts for 18.07%, while Facility (4.58%) and Others (5.02%) are small minorities.
- **Why Accuracy is Deceptive**:
  - A naive classifier predicting "Lecturer" for every input achieves **72.33% Topic Accuracy** while providing zero utility for facility or curriculum issues.
  - Macro-F1 assigns equal weight to every class regardless of support, penalizing models that fail on minority classes.
  - PhoBERT achieves substantial gains on minority classes, raising Topic Macro-F1 and Sentiment Macro-F1 significantly.

### 2.3 Compute vs Performance Trade-Offs: Engineering Reality
| Dimension | Classical Baseline | PhoBERT Multi-Task | Engineering Implication |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~0.95 MB | ~515 MB | Baseline fits in memory anywhere; PhoBERT requires ~2GB RAM / VRAM |
| **Inference Speed** | >1,000 samples/sec | ~50 samples/sec (GPU) / ~30/s (CPU) | Baseline is 25-50x faster, optimal for high-throughput batch filtering |
| **Training Time** | ~2 seconds | ~15-30 mins (GPU) / ~2-3 hours (CPU) | Baseline allows instant hyperparameter retraining |
| **Deployment Recommendation** | Microservices, embedded edge devices, low-cost CPU servers | GPU cloud endpoints where contextual nuance is critical |

---

## 3. Detailed Per-Class Breakdown

### 3.1 Sentiment Classification (3 Classes)

| Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `negative` (0) | Baseline | 88.18% | 91.06% | 89.59% | 1409 |
| `negative` (0) | PhoBERT | 94.10% | 96.17% | 95.12% | 1409 |
| `neutral` (1) | Baseline | 29.63% | 43.11% | 35.12% | 167 |
| `neutral` (1) | PhoBERT | 65.91% | 52.10% | 58.19% | 167 |
| `positive` (2) | Baseline | 94.75% | 87.48% | 90.97% | 1590 |
| `positive` (2) | PhoBERT | 95.23% | 95.47% | 95.35% | 1590 |

### 3.2 Topic Classification (4 Classes)

| Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `lecturer` (0) | Baseline | 95.81% | 81.97% | 88.35% | 2290 |
| `lecturer` (0) | PhoBERT | 94.87% | 92.01% | 93.42% | 2290 |
| `training_program` (1) | Baseline | 59.27% | 79.90% | 68.06% | 572 |
| `training_program` (1) | PhoBERT | 74.20% | 80.94% | 77.42% | 572 |
| `facility` (2) | Baseline | 90.73% | 94.48% | 92.57% | 145 |
| `facility` (2) | PhoBERT | 95.14% | 94.48% | 94.81% | 145 |
| `others` (3) | Baseline | 32.63% | 58.49% | 41.89% | 159 |
| `others` (3) | PhoBERT | 51.98% | 57.86% | 54.76% | 159 |

---

## 4. Evaluation Environment & Reproducibility Metadata
- **Evaluated Split**: `test` ($N=3,166$ samples)
- **Host Device**: `CPU`
- **Operating System**: `Windows 11 (AMD64)`
- **Python Version**: `3.12.10`
- **Evaluation CLI**: `python evaluation/evaluate.py --split test`
