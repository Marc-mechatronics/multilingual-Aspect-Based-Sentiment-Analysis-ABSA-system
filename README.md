# Arabic ABSA Pipeline

A two-stage Aspect-Based Sentiment Analysis system for Arabic restaurant and e-commerce reviews, built for a competition setting with a small, noisy dataset. Evaluated on **exact (aspect, sentiment) pair Micro-F1** — stricter than standard ABSA benchmarks.

---

## Architecture

### Stage 1 — Multi-label Aspect Detection
A One-vs-Rest logistic regression classifier predicts which of **8 aspects** (+ a "none" class) are present in each review:
`food`, `service`, `price`, `cleanliness`, `delivery`, `ambiance`, `app_experience`, `general`

### Stage 2 — Per-aspect Sentiment Classification
A separate One-vs-Rest logistic regression per aspect predicts sentiment (`positive`, `negative`, `neutral`) for each detected aspect. Aspects with only one sentiment class in training use a `ConstantSentimentModel` fallback.

---

## Feature Engineering

**170,000-dimensional sparse feature space** built from two TF-IDF vectorizers fused via `FeatureUnion`:

| Vectorizer | Analyzer | N-gram range | Max features | Notes |
|---|---|---|---|---|
| Word TF-IDF | word | 1–2 | 50,000 | `sublinear_tf=True`, `min_df=2` |
| Char TF-IDF | char_wb | 3–5 | 120,000 | `sublinear_tf=True`, captures morphology |

**Structured metadata** is appended as token suffixes to each review text before vectorization:
```
<normalized_review> business_category_restaurant platform_google_maps star_rating_4
```
This lets the model learn aspect–platform and aspect–rating correlations without a separate embedding layer.

---

## Arabic Text Normalization

All reviews pass through a custom normalization pipeline before feature extraction:

| Step | Detail |
|---|---|
| Unicode normalization | NFKC + HTML entity unescaping |
| URL / mention removal | Strips `https://...` and `@handle` tokens |
| Diacritic removal | Unicode range `\u0617–\u061A`, `\u064B–\u0652`, `\u0670` |
| Tatweel removal | `\u0640` elongation character |
| Alef unification | `أ إ آ ٱ` → `ا` |
| Ya / Waw unification | `ى` → `ي`, `ئ` → `ي`, `ؤ` → `و` |
| Repeated-char collapsing | `كككككتير` → `ككتير` (max 2 repeats) |
| Whitespace normalization | Collapse all whitespace to single space |

---

## Threshold Tuning

Default 0.5 decision thresholds are replaced by **coordinate-wise grid search** on the validation set:

- Search grid: `[0.20, 0.25, ..., 0.80]` (13 candidates per aspect)
- Strategy: optimize one aspect's threshold at a time, holding others fixed; repeat for 2 passes
- Objective: maximize Micro-F1 on exact `(aspect, sentiment)` pairs
- Result: per-aspect thresholds stored in `model.aspect_thresholds`

This is important because aspect classes are imbalanced — rare aspects (e.g. `cleanliness`, `app_experience`) benefit significantly from lower thresholds.

---

## Metric

Micro-F1 is computed on exact `(aspect, sentiment)` pairs. Both the aspect and its sentiment must match simultaneously.

```
Gold:      {("service", "positive"), ("food", "negative")}
Predicted: {("service", "positive"), ("food", "neutral")}

TP = 1  (service/positive matched)
FP = 1  (food/neutral not in gold)
FN = 1  (food/negative not predicted)

precision = 1/2 = 0.50
recall    = 1/2 = 0.50
Micro-F1  = 0.50
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

---

## Train

```bash
python train.py ^
  --train-path "C:/Users/marke/OneDrive/Desktop/DeepX/train_fixed.xlsx" ^
  --validation-path "C:/Users/marke/OneDrive/Desktop/DeepX/validation_fixed.xlsx" ^
  --output-dir artifacts
```

Training prints per-aspect sentiment classification reports and the final validation Micro-F1. The tuned model is saved to `artifacts/absa_model.joblib`.

---

## Inference

```bash
python inference.py ^
  --model-path artifacts/absa_model.joblib ^
  --test-path "C:/Users/marke/OneDrive/Desktop/DeepX/unlabeled_fixed.xlsx" ^
  --output-path submission.json
```

---

## Local Demo App

```bash
python app.py --model-path artifacts/absa_model.joblib
```

Opens `http://127.0.0.1:8000` automatically. Paste any Arabic review and optionally provide star rating, platform, and business category. The app returns detected aspects, per-aspect sentiment, aspect probabilities vs. tuned thresholds, and the exact submission JSON.

---

## Design Decisions

**Why TF-IDF + logistic regression, not a transformer?**
The dataset is small and noisy. Fine-tuning AraBERT or CAMeL-BERT on a few hundred examples risks overfitting and requires GPU resources not available in competition. The char n-gram TF-IDF (3–5) effectively captures Arabic morphological variation without stemming or a dedicated morphological analyzer. The two-stage architecture keeps aspect errors from cascading into sentiment errors uncontrolled.

**Why coordinate-wise threshold tuning?**
A joint grid search over all 9 aspects would require 13⁹ ≈ 10 billion evaluations. Coordinate-wise search with 2 passes covers 9 × 13 × 2 = 234 evaluations and empirically finds near-optimal thresholds for imbalanced multi-label problems.
