from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_utils import ASPECTS, SENTIMENTS


def to_python_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_python_types(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [to_python_types(item) for item in value]
    if isinstance(value, tuple):
        return [to_python_types(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_submission_records(records: list[dict[str, Any]]) -> None:
    if not isinstance(records, list):
        raise ValueError("Submission must be a list of records.")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be a dictionary.")

        expected_keys = {"review_id", "aspects", "aspect_sentiments"}
        actual_keys = set(record.keys())
        if actual_keys != expected_keys:
            raise ValueError(f"Record {index} must contain exactly {sorted(expected_keys)}. Found {sorted(actual_keys)}.")

        review_id = record["review_id"]
        if isinstance(review_id, np.generic):
            review_id = review_id.item()
        if not isinstance(review_id, (int, str)):
            raise ValueError(f"Record {index} review_id must be int or str. Found {type(review_id).__name__}.")

        aspects = record["aspects"]
        if not isinstance(aspects, list):
            raise ValueError(f"Record {index} aspects must be a list.")
        if not aspects:
            raise ValueError(f"Record {index} aspects must not be empty.")

        for aspect in aspects:
            if not isinstance(aspect, str):
                raise ValueError(f"Record {index} aspect values must be strings.")
            if aspect != aspect.lower():
                raise ValueError(f"Record {index} aspect values must be lowercase: {aspect!r}")
            if aspect not in ASPECTS:
                raise ValueError(f"Record {index} has invalid aspect: {aspect!r}")

        if len(set(aspects)) != len(aspects):
            raise ValueError(f"Record {index} aspects must not contain duplicates.")

        aspect_sentiments = record["aspect_sentiments"]
        if not isinstance(aspect_sentiments, dict):
            raise ValueError(f"Record {index} aspect_sentiments must be a dictionary.")

        sentiment_keys = set(aspect_sentiments.keys())
        aspect_keys = set(aspects)
        if sentiment_keys != aspect_keys:
            raise ValueError(
                f"Record {index} aspect_sentiments keys must match aspects exactly. "
                f"Expected {sorted(aspect_keys)}, found {sorted(sentiment_keys)}."
            )

        for aspect, sentiment in aspect_sentiments.items():
            if aspect != aspect.lower():
                raise ValueError(f"Record {index} aspect_sentiments key must be lowercase: {aspect!r}")
            if aspect not in ASPECTS:
                raise ValueError(f"Record {index} has invalid aspect key: {aspect!r}")
            if not isinstance(sentiment, str):
                raise ValueError(f"Record {index} sentiment for {aspect!r} must be a string.")
            if sentiment != sentiment.lower():
                raise ValueError(f"Record {index} sentiment values must be lowercase: {sentiment!r}")
            if sentiment not in SENTIMENTS:
                raise ValueError(f"Record {index} has invalid sentiment: {sentiment!r}")


def save_submission(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    serializable_records = to_python_types(records)
    validate_submission_records(serializable_records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(serializable_records, handle, ensure_ascii=False, indent=2)
    return destination

