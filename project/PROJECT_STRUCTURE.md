# Project Structure

## Directory Tree

```text
C:\Users\marke\Documents\Codex\2026-04-24\files-mentioned-by-the-user-unlabeled
|   app.py
|   data_utils.py
|   inference.py
|   model.py
|   README.md
|   requirements.txt
|   submission.json
|   submission_utils.py
|   train.py
|   validation_submission_preview.json
|
+---artifacts
|       absa_model.joblib
|       dataset_summaries.json
|       validation_metrics.json
|
\---__pycache__
        app.cpython-312.pyc
        data_utils.cpython-312.pyc
        inference.cpython-312.pyc
        model.cpython-312.pyc
        submission_utils.cpython-312.pyc
        train.cpython-312.pyc
```

## Modularity Map

- `data_utils.py`
  Handles Excel loading, cleaning, normalization, schema standardization, inference dataframe preparation, and Micro-F1 computation on `(aspect, sentiment)` pairs.

- `model.py`
  Contains the two-stage ABSA pipeline:
  1. multi-label aspect detection
  2. per-aspect sentiment classification

- `train.py`
  Loads train and validation files, trains the model, tunes aspect thresholds, evaluates on validation, and saves artifacts.

- `inference.py`
  Loads the saved model and generates predictions in the required competition submission format.

- `submission_utils.py`
  Validates the final submission schema and saves valid JSON with UTF-8 encoding.

- `app.py`
  Local lightweight browser app for interactive prediction. You paste a review and it shows detected aspects, sentiments, scores, and output JSON.

- `artifacts/absa_model.joblib`
  Saved trained model used by inference and the app.

- `artifacts/dataset_summaries.json`
  Summary statistics for train and validation datasets.

- `artifacts/validation_metrics.json`
  Validation metrics including Micro-F1 and the tuned aspect thresholds.

- `submission.json`
  Final generated competition submission for the unlabeled test file.

- `validation_submission_preview.json`
  Validation-set predictions produced through the standalone inference pipeline for end-to-end checking.

## Flow

```text
train.py
  -> data_utils.py
  -> model.py
  -> artifacts/absa_model.joblib

inference.py
  -> data_utils.py
  -> model.py
  -> submission_utils.py
  -> submission.json

app.py
  -> data_utils.py
  -> model.py
  -> interactive prediction result
```

## Main Run Commands

```bash
python train.py --train-path "C:/Users/marke/OneDrive/Desktop/DeepX/train_fixed.xlsx" --validation-path "C:/Users/marke/OneDrive/Desktop/DeepX/validation_fixed.xlsx" --output-dir artifacts
```

```bash
python inference.py --model-path artifacts/absa_model.joblib --test-path "C:/Users/marke/OneDrive/Desktop/DeepX/unlabeled_fixed.xlsx" --output-path submission.json
```

```bash
python app.py --model-path artifacts/absa_model.joblib
```
