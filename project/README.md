# Arabic ABSA Pipeline

This workspace contains a simple two-stage Aspect-Based Sentiment Analysis pipeline for the competition:

1. Multi-label aspect detection
2. Per-aspect sentiment classification

The model is intentionally lightweight and production-ready for a small, noisy Arabic dataset:

- Word + character TF-IDF features
- One-vs-rest logistic regression for aspect detection
- Per-aspect one-vs-rest logistic regression for sentiment
- Threshold tuning on validation to optimize Micro-F1 over `(aspect, sentiment)` pairs

## Files

- `data_utils.py`: loading, cleaning, normalization, dataset summaries, Micro-F1
- `model.py`: two-stage ABSA model and threshold tuning
- `train.py`: training + validation + artifact saving
- `inference.py`: test inference + `submission.json` generation
- `app.py`: lightweight local web app to paste a review and see the prediction
- `submission_utils.py`: submission schema validation + JSON saving

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

## Inference

```bash
python inference.py ^
  --model-path artifacts/absa_model.joblib ^
  --test-path "C:/Users/marke/OneDrive/Desktop/DeepX/unlabeled_fixed.xlsx" ^
  --output-path submission.json
```

## App

```bash
python app.py --model-path artifacts/absa_model.joblib
```

Then open `http://127.0.0.1:8000` if the browser does not open automatically.

## Metric

Micro-F1 is computed on exact `(aspect, sentiment)` pairs.

Example:

- Gold: `("service", "positive")`, `("food", "negative")`
- Prediction: `("service", "positive")`, `("food", "neutral")`

This yields:

- `TP = 1`
- `FP = 1`
- `FN = 1`

Then:

- `precision = TP / (TP + FP)`
- `recall = TP / (TP + FN)`
- `micro_f1 = 2 * precision * recall / (precision + recall)`
