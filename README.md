# Arabic ABSA Pipeline

A two-stage Aspect-Based Sentiment Analysis system for Arabic restaurant and e-commerce reviews, built for a competition setting. Evaluated on **exact (aspect, sentiment) pair Micro-F1** — stricter than standard ABSA benchmarks, requiring both the aspect and its sentiment to match simultaneously.

---

## Results

| Metric | Score |
|---|---|
| **Micro-F1** | **0.730** |
| Micro-Precision | 0.723 |
| Micro-Recall | 0.738 |
| TP / FP / FN | 620 / 238 / 220 |

**Per-aspect sentiment accuracy (validation set):**

| Aspect | Weighted F1 | Support | Notes |
|---|---|---|---|
| service | 0.968 | 253 | Strongest — largest support |
| delivery | 0.949 | 48 | Near-perfect on neg/pos |
| app_experience | 0.862 | 120 | Neutral class struggles |
| ambiance | 0.886 | 100 | |
| general | 0.865 | 74 | Small neg/neutral support |
| price | 0.857 | 80 | |
| cleanliness | 0.827 | 51 | Smallest support |
| food | 0.826 | 102 | Neutral class (9 samples) fails |

**Neutral sentiment** is the hardest class across all aspects due to severe imbalance (149 neutral vs 1,646 positive and 1,538 negative in training).

---

## Dataset

| Split | Reviews | Duplicate texts | Aspects (total labels) |
|---|---|---|---|
| Train | 1,971 | 88 | 3,277 |
| Validation | 500 | 15 | 840 |

**Aspect distribution (train):** service (988) and app_experience (453) dominate; delivery (161) and cleanliness (185) are the rarest — driving threshold tuning decisions.

**Label cardinality:** 54% of reviews have a single aspect; 28% have two; 14% have three or more. Maximum observed: 6 aspects in one review.

---

## Architecture

### Stage 1 — Multi-label Aspect Detection
One-vs-Rest logistic regression predicts which of **8 aspects** (+ a "none" class) are present:
`food`, `service`, `price`, `cleanliness`, `delivery`, `ambiance`, `app_experience`, `general`

### Stage 2 — Per-aspect Sentiment Classification
A separate One-vs-Rest logistic regression per aspect predicts sentiment (`positive`, `negative`, `neutral`) for each detected aspect. Aspects with only one sentiment class in training use a `ConstantSentimentModel` fallback to avoid fitting noise.

---

## Feature Engineering

**170,000-dimensional sparse feature space** from two TF-IDF vectorizers fused via `FeatureUnion`:

| Vectorizer | Analyzer | N-gram range | Max features | Notes |
|---|---|---|---|---|
| Word TF-IDF | word | 1–2 | 50,000 | `sublinear_tf=True`, `min_df=2` |
| Char TF-IDF | char_wb | 3–5 | 120,000 | Captures Arabic morphology |

**Structured metadata** is appended as token suffixes before vectorization:
```
<normalized_review> business_category_restaurant platform_google_maps star_rating_4
```
This lets a bag-of-words model exploit platform and rating signals without a separate embedding layer.

---

## Arabic Text Normalization

| Step | Detail |
|---|---|
| Unicode normalization | NFKC + HTML entity unescaping |
| URL / mention removal | Strips `https://...` and `@handle` tokens |
| Diacritic removal | Unicode range `\u0617–\u061A`, `\u064B–\u0652`, `\u0670` |
| Tatweel removal | `\u0640` elongation character |
| Alef unification | `أ إ آ ٱ` → `ا` |
| Ya / Waw unification | `ى` → `ي`, `ئ` → `ي`, `ؤ` → `و` |
| Repeated-char collapsing | Max 2 consecutive repeats (e.g. `كككككتير` → `ككتير`) |
| Whitespace normalization | Collapse all whitespace to single space |

---

## Threshold Tuning

Default 0.5 thresholds are replaced by **coordinate-wise grid search** on the validation set, directly optimizing Micro-F1:

| Aspect | Tuned threshold | Direction from 0.5 |
|---|---|---|
| food | 0.70 | ↑ higher — avoids false positives |
| general | 0.75 | ↑ highest — very conservative |
| ambiance | 0.60 | ↑ slightly |
| cleanliness | 0.65 | ↑ higher |
| price | 0.50 | unchanged |
| service | 0.40 | ↓ lower — maximize recall on largest class |
| app_experience | 0.40 | ↓ lower |
| delivery | 0.45 | ↓ slightly |
| none | 0.40 | ↓ lower |

Search grid: `[0.20, 0.25, ..., 0.80]` (13 candidates × 9 aspects × 2 passes = 234 evaluations). A joint grid search would require 13⁹ ≈ 10 billion evaluations.

---

## Metric

Micro-F1 on exact `(aspect, sentiment)` pairs. Both must match simultaneously.

```
Gold:      {("service", "positive"), ("food", "negative")}
Predicted: {("service", "positive"), ("food", "neutral")}

TP = 1,  FP = 1,  FN = 1
precision = 0.50,  recall = 0.50,  Micro-F1 = 0.50
```

---

## Files

| File | Purpose |
|---|---|
| `data_utils.py` | Loading, Arabic normalization, column standardization, dataset summaries, Micro-F1 |
| `model.py` | Two-stage ABSA model, threshold tuning, evaluation |
| `train.py` | Training + validation + artifact saving |
| `inference.py` | Test inference + `submission.json` generation |
| `app.py` | Lightweight local web demo (stdlib only, no framework) |
| `submission_utils.py` | Submission schema validation + JSON saving |

---

## Install

```bash
python -m pip install -r requirements.txt
```

## Train

```bash
python train.py ^
  --train-path "C:/Users/marke/OneDrive/Desktop/DeepX/train_fixed.xlsx" ^
  --validation-path "C:/Users/marke/OneDrive/Desktop/DeepX/validation_fixed.xlsx" ^
  --output-dir artifacts
```

Saves `artifacts/absa_model.joblib`, `artifacts/validation_metrics.json`, and `artifacts/dataset_summaries.json`.

## Inference

```bash
python inference.py ^
  --model-path artifacts/absa_model.joblib ^
  --test-path "C:/Users/marke/OneDrive/Desktop/DeepX/unlabeled_fixed.xlsx" ^
  --output-path submission.json
```

## Local Demo App

```bash
python app.py --model-path artifacts/absa_model.joblib
```

Opens `http://127.0.0.1:8000`. Paste any Arabic review with optional star rating, platform, and business category. Returns detected aspects, per-aspect sentiment, aspect probabilities vs. tuned thresholds, and the exact submission JSON.

---

## Design Decisions

**Why TF-IDF + logistic regression over AraBERT/CAMeL-BERT?**
At 1,971 training examples, fine-tuning a transformer risks overfitting and requires GPU resources unavailable in this competition setting. Char n-gram TF-IDF (3–5) handles Arabic morphological variation effectively without a stemmer or morphological analyzer, and the two-stage design prevents aspect detection errors from uncontrollably cascading into sentiment errors.

**Why coordinate-wise threshold tuning?**
Rare aspects (cleanliness: 185 train examples, delivery: 161) need higher thresholds to avoid false positives, while dominant aspects (service: 988) benefit from lower thresholds to maximize recall. A single global threshold cannot handle this imbalance.
