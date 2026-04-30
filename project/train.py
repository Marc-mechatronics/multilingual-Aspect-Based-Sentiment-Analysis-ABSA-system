from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import load_absa_excel, seed_everything, summarize_dataset
from model import TwoStageABSAModel


DEFAULT_TRAIN_PATH = Path(r"C:/Users/marke/OneDrive/Desktop/DeepX/train_fixed.xlsx")
DEFAULT_VALIDATION_PATH = Path(r"C:/Users/marke/OneDrive/Desktop/DeepX/validation_fixed.xlsx")
DEFAULT_OUTPUT_DIR = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-stage Arabic ABSA pipeline.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH, help="Path to the training Excel file.")
    parser.add_argument(
        "--validation-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Path to the validation Excel file.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save artifacts.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--word-min-df", type=int, default=2, help="Minimum document frequency for TF-IDF features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.random_state)

    train_df = load_absa_excel(args.train_path, labeled=True)
    validation_df = load_absa_excel(args.validation_path, labeled=True)

    train_summary = summarize_dataset(train_df, labeled=True)
    validation_summary = summarize_dataset(validation_df, labeled=True)

    model = TwoStageABSAModel(random_state=args.random_state, word_min_df=args.word_min_df)
    model.fit(train_df, validation_df=validation_df)
    metrics = model.evaluate(validation_df)

    model_path = model.save(args.output_dir)
    metrics_path = args.output_dir / "validation_metrics.json"
    summaries_path = args.output_dir / "dataset_summaries.json"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summaries_path.open("w", encoding="utf-8") as handle:
        json.dump({"train": train_summary, "validation": validation_summary}, handle, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(json.dumps({"model_path": str(model_path), "metrics_path": str(metrics_path), "micro_f1": metrics["micro_f1"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

