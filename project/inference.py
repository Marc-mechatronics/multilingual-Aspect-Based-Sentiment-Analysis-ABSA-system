from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import build_gold_records, load_absa_excel, micro_f1_on_pairs
from model import TwoStageABSAModel
from submission_utils import save_submission, validate_submission_records


DEFAULT_TEST_PATH = Path(r"C:/Users/marke/OneDrive/Desktop/DeepX/unlabeled_fixed.xlsx")
DEFAULT_MODEL_PATH = Path("artifacts/absa_model.joblib")
DEFAULT_OUTPUT_PATH = Path("submission.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the Arabic ABSA pipeline.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained model artifact.")
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH, help="Path to the test Excel file.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to save submission.json.")
    parser.add_argument(
        "--labeled",
        action="store_true",
        help="Set this flag if the input file includes gold aspects/aspect_sentiments and you want an evaluation score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = TwoStageABSAModel.load(args.model_path)
    df = load_absa_excel(args.test_path, labeled=args.labeled)

    records = model.predict_records(df)
    validate_submission_records(records)
    output_path = save_submission(records, args.output_path)

    output = {"output_path": str(output_path), "records": len(records)}
    if args.labeled:
        metrics = micro_f1_on_pairs(build_gold_records(df), records)
        output["micro_f1"] = metrics["micro_f1"]

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

