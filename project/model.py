from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from data_utils import ASPECTS, SENTIMENTS, build_aspect_target_matrix, build_gold_records, micro_f1_on_pairs


def build_feature_union(word_min_df: int = 2) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=word_min_df,
                    max_features=50000,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=word_min_df,
                    max_features=120000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def ensure_writable_matrix(matrix: Any) -> Any:
    if sparse.issparse(matrix):
        return matrix.tocsr(copy=True)
    return np.asarray(matrix).copy()


def build_aspect_pipeline(random_state: int = 42, word_min_df: int = 2) -> Pipeline:
    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2500,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        ),
        n_jobs=1,
    )
    return Pipeline(
        [
            ("features", build_feature_union(word_min_df=word_min_df)),
            ("make_writable", FunctionTransformer(ensure_writable_matrix, accept_sparse=True)),
            ("classifier", classifier),
        ]
    )


def build_sentiment_pipeline(random_state: int = 42, word_min_df: int = 2) -> Pipeline:
    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2500,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        ),
        n_jobs=1,
    )
    return Pipeline(
        [
            ("features", build_feature_union(word_min_df=word_min_df)),
            ("make_writable", FunctionTransformer(ensure_writable_matrix, accept_sparse=True)),
            ("classifier", classifier),
        ]
    )


class ConstantSentimentModel:
    def __init__(self, label: str) -> None:
        self.label = label

    def fit(self, texts: list[str], labels: list[str] | None = None) -> "ConstantSentimentModel":
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.label] * len(texts), dtype=object)


@dataclass
class TwoStageABSAModel:
    aspects: list[str] = field(default_factory=lambda: list(ASPECTS))
    sentiments: list[str] = field(default_factory=lambda: list(SENTIMENTS))
    random_state: int = 42
    word_min_df: int = 2
    aspect_model: Pipeline | None = None
    sentiment_models: dict[str, Any] = field(default_factory=dict)
    aspect_thresholds: dict[str, float] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame, validation_df: pd.DataFrame | None = None) -> "TwoStageABSAModel":
        train_texts = train_df["model_text"].tolist()
        aspect_targets = build_aspect_target_matrix(train_df, self.aspects)

        self.aspect_model = build_aspect_pipeline(random_state=self.random_state, word_min_df=self.word_min_df)
        self.aspect_model.fit(train_texts, aspect_targets)

        self.sentiment_models = {}
        for aspect in self.aspects:
            if aspect == "none":
                self.sentiment_models[aspect] = ConstantSentimentModel("neutral")
                continue

            subset = train_df[train_df["aspects"].apply(lambda values: aspect in values)]
            texts = subset["model_text"].tolist()
            labels = subset["aspect_sentiments"].apply(lambda mapping: mapping[aspect]).tolist()

            if not texts:
                self.sentiment_models[aspect] = ConstantSentimentModel("neutral")
                continue

            if len(set(labels)) == 1:
                self.sentiment_models[aspect] = ConstantSentimentModel(labels[0])
                continue

            model = build_sentiment_pipeline(random_state=self.random_state, word_min_df=self.word_min_df)
            model.fit(texts, labels)
            self.sentiment_models[aspect] = model

        self.aspect_thresholds = {aspect: 0.5 for aspect in self.aspects}
        if validation_df is not None and len(validation_df) > 0:
            self.aspect_thresholds = self.tune_aspect_thresholds(validation_df)

        return self

    def predict_aspect_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.aspect_model is None:
            raise ValueError("Model has not been fitted yet.")
        probabilities = self.aspect_model.predict_proba(df["model_text"].tolist())
        return pd.DataFrame(probabilities, columns=self.aspects, index=df.index)

    def predict_sentiments_for_dataframe(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        texts = df["model_text"].tolist()
        predictions: dict[str, np.ndarray] = {}
        for aspect, model in self.sentiment_models.items():
            predictions[aspect] = model.predict(texts)
        return predictions

    def _decode_aspects(self, probability_row: pd.Series, thresholds: dict[str, float] | None = None) -> list[str]:
        active_thresholds = thresholds or self.aspect_thresholds
        selected = [aspect for aspect in self.aspects if probability_row[aspect] >= active_thresholds.get(aspect, 0.5)]

        non_none_aspects = [aspect for aspect in selected if aspect != "none"]
        if non_none_aspects:
            return [aspect for aspect in self.aspects if aspect in non_none_aspects]

        none_probability = float(probability_row["none"]) if "none" in probability_row else 0.0
        best_non_none = max(
            (aspect for aspect in self.aspects if aspect != "none"),
            key=lambda aspect: float(probability_row[aspect]),
        )
        best_non_none_probability = float(probability_row[best_non_none])

        if none_probability >= active_thresholds.get("none", 0.5) or none_probability >= best_non_none_probability:
            return ["none"]
        return [best_non_none]

    def predict_records(self, df: pd.DataFrame, thresholds: dict[str, float] | None = None) -> list[dict[str, Any]]:
        probability_df = self.predict_aspect_probabilities(df)
        sentiment_predictions = self.predict_sentiments_for_dataframe(df)
        records: list[dict[str, Any]] = []

        for row_index, (_, row) in enumerate(df.iterrows()):
            selected_aspects = self._decode_aspects(probability_df.loc[row.name], thresholds=thresholds)
            aspect_sentiments = {
                aspect: ("neutral" if aspect == "none" else str(sentiment_predictions[aspect][row_index]))
                for aspect in selected_aspects
            }
            records.append(
                {
                    "review_id": row["review_id"],
                    "aspects": selected_aspects,
                    "aspect_sentiments": aspect_sentiments,
                }
            )
        return records

    def tune_aspect_thresholds(
        self,
        validation_df: pd.DataFrame,
        grid: list[float] | None = None,
        passes: int = 2,
    ) -> dict[str, float]:
        search_grid = grid or [round(value, 2) for value in np.arange(0.2, 0.81, 0.05)]
        thresholds = {aspect: 0.5 for aspect in self.aspects}
        gold_records = build_gold_records(validation_df)
        probability_df = self.predict_aspect_probabilities(validation_df)
        sentiment_predictions = self.predict_sentiments_for_dataframe(validation_df)

        best_score = self._score_thresholds(validation_df, gold_records, probability_df, sentiment_predictions, thresholds)

        for _ in range(passes):
            improved = False
            for aspect in self.aspects:
                best_aspect_threshold = thresholds[aspect]
                best_aspect_score = best_score
                for candidate in search_grid:
                    trial_thresholds = dict(thresholds)
                    trial_thresholds[aspect] = candidate
                    trial_score = self._score_thresholds(
                        validation_df,
                        gold_records,
                        probability_df,
                        sentiment_predictions,
                        trial_thresholds,
                    )
                    if trial_score > best_aspect_score:
                        best_aspect_score = trial_score
                        best_aspect_threshold = candidate
                if best_aspect_threshold != thresholds[aspect]:
                    thresholds[aspect] = best_aspect_threshold
                    best_score = best_aspect_score
                    improved = True
            if not improved:
                break

        return thresholds

    def _score_thresholds(
        self,
        validation_df: pd.DataFrame,
        gold_records: list[dict[str, Any]],
        probability_df: pd.DataFrame,
        sentiment_predictions: dict[str, np.ndarray],
        thresholds: dict[str, float],
    ) -> float:
        predicted_records: list[dict[str, Any]] = []

        for row_index, (_, row) in enumerate(validation_df.iterrows()):
            selected_aspects = self._decode_aspects(probability_df.loc[row.name], thresholds=thresholds)
            predicted_records.append(
                {
                    "review_id": row["review_id"],
                    "aspects": selected_aspects,
                    "aspect_sentiments": {
                        aspect: ("neutral" if aspect == "none" else str(sentiment_predictions[aspect][row_index]))
                        for aspect in selected_aspects
                    },
                }
            )

        return float(micro_f1_on_pairs(gold_records, predicted_records)["micro_f1"])

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        gold_records = build_gold_records(df)
        predicted_records = self.predict_records(df)
        metrics = micro_f1_on_pairs(gold_records, predicted_records)

        sentiment_reports: dict[str, Any] = {}
        for aspect in self.aspects:
            if aspect == "none":
                continue
            subset = df[df["aspects"].apply(lambda values: aspect in values)]
            if subset.empty:
                continue
            gold_labels = subset["aspect_sentiments"].apply(lambda mapping: mapping[aspect]).tolist()
            predicted_labels = self.sentiment_models[aspect].predict(subset["model_text"].tolist()).tolist()
            sentiment_reports[aspect] = classification_report(
                gold_labels,
                predicted_labels,
                output_dict=True,
                zero_division=0,
            )

        metrics["aspect_thresholds"] = self.aspect_thresholds
        metrics["sentiment_reports"] = sentiment_reports
        return metrics

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / "absa_model.joblib"
        joblib.dump(self, model_path)
        return model_path

    @classmethod
    def load(cls, model_path: str | Path) -> "TwoStageABSAModel":
        return joblib.load(model_path)
