import argparse
import json
import sys
from pathlib import Path

import pandas as pd



# Utility helpers

def _load_csv(path: Path, expected_columns=None, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=usecols)
    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {', '.join(missing)}"
            )
    # Drop unnamed index columns that sneak in from `to_csv(index=True)`
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df



# 1. Training

def train_average_baseline(train_csv: Path, model_path: Path) -> None:
    df = _load_csv(train_csv, expected_columns=["carb"])
    # Force numeric & drop rows where parsing failed
    df["carb"] = pd.to_numeric(df["carb"], errors="coerce")
    df = df.dropna(subset=["carb"])
    if df.empty:
        raise RuntimeError("No valid rows with numeric `carb` after cleaning.")

    mean_carb = df["carb"].mean()
    model_path.write_text(json.dumps({"mean_carb": mean_carb}, indent=2))
    print(f"Saved model with mean={mean_carb:.4f} g ->  {model_path}")



# 2. Prediction

def predict_average_baseline(
    model_path: Path, test_csv: Path, output_csv: Path
) -> None:
    model = json.loads(model_path.read_text())
    if "mean_carb" not in model:
        raise ValueError(f"{model_path} is malformed (missing `mean_carb`).")

    df_test = _load_csv(test_csv)
    df_test["carb"] = model["mean_carb"]  # vectorized broadcast
    df_test.to_csv(output_csv, index=False)
    print(f"Wrote predictions ->  {output_csv}  "
          f"(constant {model['mean_carb']:.4f} g)")



# 3. Command-line interface

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Average-baseline trainer & predictor for NutriBench."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Compute mean carb on training set")
    t.add_argument("train_csv", type=Path)
    t.add_argument("model_json", type=Path)

    pr = sub.add_parser("predict", help="Generate predictions on test/val set")
    pr.add_argument("model_json", type=Path)
    pr.add_argument("test_csv", type=Path)
    pr.add_argument("output_csv", type=Path)
    return p


def main(argv=None):
    args = _build_cli().parse_args(argv)

    if args.cmd == "train":
        train_average_baseline(args.train_csv, args.model_json)
    elif args.cmd == "predict":
        predict_average_baseline(args.model_json, args.test_csv, args.output_csv)
    else:
        raise ValueError(f"Unknown sub-command {args.cmd}")


if __name__ == "__main__":  # Allows both CLI use & import in notebooks
    main()
