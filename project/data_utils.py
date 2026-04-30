from __future__ import annotations

import ast
import html
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ASPECTS = [
    "food",
    "service",
    "price",
    "cleanliness",
    "delivery",
    "ambiance",
    "app_experience",
    "general",
    "none",
]

SENTIMENTS = ["positive", "negative", "neutral"]

REQUIRED_BASE_COLUMNS = ["review_id", "review_text"]
OPTIONAL_COLUMNS = ["star_rating", "date", "business_name", "business_category", "platform"]
REQUIRED_LABELED_COLUMNS = ["aspects", "aspect_sentiments"]
CANONICAL_COLUMNS = REQUIRED_BASE_COLUMNS + OPTIONAL_COLUMNS + REQUIRED_LABELED_COLUMNS

COLUMN_ALIASES = {
    "review_id": "review_id",
    "id": "review_id",
    "reviewid": "review_id",
    "review_text": "review_text",
    "text": "review_text",
    "review": "review_text",
    "sentence": "review_text",
    "content": "review_text",
    "star_rating": "star_rating",
    "rating": "star_rating",
    "stars": "star_rating",
    "date": "date",
    "business_name": "business_name",
    "store_name": "business_name",
    "merchant_name": "business_name",
    "business_category": "business_category",
    "category": "business_category",
    "domain": "business_category",
    "platform": "platform",
    "source": "platform",
    "channel": "platform",
    "aspects": "aspects",
    "aspect": "aspects",
    "labels": "aspects",
    "aspect_sentiments": "aspect_sentiments",
    "sentiments": "aspect_sentiments",
    "aspect_sentiment": "aspect_sentiments",
}

ASPECT_ALIASES = {
    "app experience": "app_experience",
    "app-experience": "app_experience",
    "app_experience": "app_experience",
}

SENTIMENT_ALIASES = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "negative": "negative",
    "neu": "neutral",
    "neutral": "neutral",
}

ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
ARABIC_TATWEEL_RE = re.compile(r"\u0640")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_text_key(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def standardize_columns(df: pd.DataFrame, labeled: bool | None = None) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in df.columns:
        normalized = normalize_text_key(column)
        if normalized in COLUMN_ALIASES:
            rename_map[column] = COLUMN_ALIASES[normalized]
    standardized = df.rename(columns=rename_map).copy()

    missing_base = [column for column in REQUIRED_BASE_COLUMNS if column not in standardized.columns]
    if missing_base:
        raise ValueError(f"Missing required columns: {missing_base}")

    inferred_labeled = labeled
    if inferred_labeled is None:
        inferred_labeled = all(column in standardized.columns for column in REQUIRED_LABELED_COLUMNS)

    if inferred_labeled:
        missing_labeled = [column for column in REQUIRED_LABELED_COLUMNS if column not in standardized.columns]
        if missing_labeled:
            raise ValueError(f"Missing labeled columns: {missing_labeled}")

    for column in OPTIONAL_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = None

    return standardized


def normalize_arabic_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = html.unescape(normalized)
    normalized = normalized.replace("\r", " ").replace("\n", " ")
    normalized = URL_RE.sub(" ", normalized)
    normalized = MENTION_RE.sub(" ", normalized)
    normalized = ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = ARABIC_TATWEEL_RE.sub("", normalized)
    normalized = (
        normalized.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    normalized = REPEATED_CHAR_RE.sub(r"\1\1", normalized)
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def normalize_meta_token(value: Any, default: str = "unknown") -> str:
    text = normalize_arabic_text(value)
    if not text:
        return default
    text = text.lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\u0600-\u06FF]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def parse_json_like(value: Any, expected_type: type, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, expected_type):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                if isinstance(parsed, expected_type):
                    return parsed
            except Exception:
                continue
    raise ValueError(f"Could not parse value as {expected_type.__name__}: {value!r}")


def normalize_aspect_label(aspect: Any) -> str:
    normalized = normalize_arabic_text(aspect).lower().replace("-", "_").replace(" ", "_")
    normalized = ASPECT_ALIASES.get(normalized, normalized)
    if normalized not in ASPECTS:
        raise ValueError(f"Invalid aspect label: {aspect!r}")
    return normalized


def normalize_sentiment_label(sentiment: Any) -> str:
    normalized = normalize_arabic_text(sentiment).lower().replace("-", "_").replace(" ", "_")
    normalized = SENTIMENT_ALIASES.get(normalized, normalized)
    if normalized not in SENTIMENTS:
        raise ValueError(f"Invalid sentiment label: {sentiment!r}")
    return normalized


def normalize_aspects_list(aspects: Any) -> list[str]:
    parsed = parse_json_like(aspects, list, [])
    normalized: list[str] = []
    seen: set[str] = set()
    for aspect in parsed:
        canonical = normalize_aspect_label(aspect)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    if "none" in seen and len(normalized) > 1:
        normalized = [aspect for aspect in normalized if aspect != "none"]
    return normalized


def normalize_aspect_sentiments(aspect_sentiments: Any) -> dict[str, str]:
    parsed = parse_json_like(aspect_sentiments, dict, {})
    normalized: dict[str, str] = {}
    for aspect, sentiment in parsed.items():
        canonical_aspect = normalize_aspect_label(aspect)
        canonical_sentiment = normalize_sentiment_label(sentiment)
        normalized[canonical_aspect] = canonical_sentiment
    if "none" in normalized and len(normalized) > 1:
        normalized = {aspect: sentiment for aspect, sentiment in normalized.items() if aspect != "none"}
    return normalized


def build_model_text(row: pd.Series) -> str:
    review_text = row["normalized_review_text"]
    category = normalize_meta_token(row.get("business_category"))
    platform = normalize_meta_token(row.get("platform"))
    rating_value = to_python_scalar(row.get("star_rating"))
    if rating_value is None:
        rating = "unknown"
    else:
        try:
            rating = str(int(float(rating_value)))
        except Exception:
            rating = normalize_meta_token(rating_value)

    parts = [
        review_text,
        f"business_category_{category}",
        f"platform_{platform}",
        f"star_rating_{rating}",
    ]
    return " ".join(part for part in parts if part).strip()


def validate_record_schema(aspects: list[str], aspect_sentiments: dict[str, str]) -> None:
    aspect_set = set(aspects)
    dict_keys = set(aspect_sentiments.keys())
    if aspect_set != dict_keys:
        raise ValueError(
            f"Aspect list and aspect_sentiments keys do not match. aspects={aspects}, keys={sorted(dict_keys)}"
        )


def load_absa_excel(path: str | Path, labeled: bool | None = None, sheet_name: int | str = 0) -> pd.DataFrame:
    raw_df = pd.read_excel(path, sheet_name=sheet_name)
    df = standardize_columns(raw_df, labeled=labeled)

    inferred_labeled = labeled
    if inferred_labeled is None:
        inferred_labeled = all(column in df.columns for column in REQUIRED_LABELED_COLUMNS)

    df["review_id"] = df["review_id"].apply(to_python_scalar)
    df["review_text"] = df["review_text"].fillna("").astype(str)
    df["normalized_review_text"] = df["review_text"].apply(normalize_arabic_text)

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = None

    if inferred_labeled:
        df["aspects"] = df["aspects"].apply(normalize_aspects_list)
        df["aspect_sentiments"] = df["aspect_sentiments"].apply(normalize_aspect_sentiments)
        for aspects, aspect_sentiments in zip(df["aspects"], df["aspect_sentiments"]):
            validate_record_schema(aspects, aspect_sentiments)
    else:
        df["aspects"] = [[] for _ in range(len(df))]
        df["aspect_sentiments"] = [{} for _ in range(len(df))]

    df["model_text"] = df.apply(build_model_text, axis=1)
    return df[REQUIRED_BASE_COLUMNS + OPTIONAL_COLUMNS + ["aspects", "aspect_sentiments", "normalized_review_text", "model_text"]]


def prepare_inference_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = {column: None for column in REQUIRED_BASE_COLUMNS + OPTIONAL_COLUMNS}
        row["review_id"] = to_python_scalar(record.get("review_id", index + 1))
        row["review_text"] = str(record.get("review_text", "") or "")
        for column in OPTIONAL_COLUMNS:
            if column in record:
                row[column] = record[column]
        rows.append(row)

    df = pd.DataFrame(rows, columns=REQUIRED_BASE_COLUMNS + OPTIONAL_COLUMNS)
    df["normalized_review_text"] = df["review_text"].apply(normalize_arabic_text)
    df["aspects"] = [[] for _ in range(len(df))]
    df["aspect_sentiments"] = [{} for _ in range(len(df))]
    df["model_text"] = df.apply(build_model_text, axis=1)
    return df[REQUIRED_BASE_COLUMNS + OPTIONAL_COLUMNS + ["aspects", "aspect_sentiments", "normalized_review_text", "model_text"]]


def summarize_dataset(df: pd.DataFrame, labeled: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_review_ids": int(df.duplicated(subset=["review_id"]).sum()),
        "duplicate_review_texts": int(df.duplicated(subset=["review_text"]).sum()),
        "empty_review_texts": int((df["normalized_review_text"].str.len() == 0).sum()),
    }

    if labeled:
        aspect_counts = {aspect: 0 for aspect in ASPECTS}
        sentiment_counts = {sentiment: 0 for sentiment in SENTIMENTS}
        pair_counts: dict[str, int] = {}
        cardinality_counts: dict[int, int] = {}

        for _, row in df.iterrows():
            aspects = row["aspects"]
            aspect_sentiments = row["aspect_sentiments"]
            cardinality = len(aspects)
            cardinality_counts[cardinality] = cardinality_counts.get(cardinality, 0) + 1
            for aspect in aspects:
                aspect_counts[aspect] += 1
                sentiment = aspect_sentiments[aspect]
                sentiment_counts[sentiment] += 1
                pair_key = f"{aspect}:{sentiment}"
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        summary["aspect_counts"] = aspect_counts
        summary["sentiment_counts"] = sentiment_counts
        summary["aspect_cardinality"] = dict(sorted(cardinality_counts.items()))
        summary["top_pairs"] = dict(sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)[:15])

    return summary


def build_aspect_target_matrix(df: pd.DataFrame, aspects: list[str] | None = None) -> np.ndarray:
    aspect_order = aspects or ASPECTS
    matrix = np.zeros((len(df), len(aspect_order)), dtype=np.int32)
    for row_index, review_aspects in enumerate(df["aspects"]):
        aspect_lookup = set(review_aspects)
        for aspect_index, aspect in enumerate(aspect_order):
            matrix[row_index, aspect_index] = int(aspect in aspect_lookup)
    return matrix


def build_gold_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "review_id": to_python_scalar(row["review_id"]),
                "aspects": list(row["aspects"]),
                "aspect_sentiments": dict(row["aspect_sentiments"]),
            }
        )
    return records


def record_to_pair_set(record: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    aspects = record["aspects"]
    aspect_sentiments = record["aspect_sentiments"]
    for aspect in aspects:
        pairs.add((aspect, aspect_sentiments[aspect]))
    return pairs


def micro_f1_on_pairs(gold_records: list[dict[str, Any]], pred_records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(gold_records) != len(pred_records):
        raise ValueError("Gold and prediction sizes do not match.")

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold_record, pred_record in zip(gold_records, pred_records):
        gold_pairs = record_to_pair_set(gold_record)
        pred_pairs = record_to_pair_set(pred_record)
        true_positive += len(gold_pairs & pred_pairs)
        false_positive += len(pred_pairs - gold_pairs)
        false_negative += len(gold_pairs - pred_pairs)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    micro_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": micro_f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }
