"""Evaluate the frozen production artifact on the untouched hold-out test set.

    python eval.py                                  # default: data/splits/holdout_test.csv
    python eval.py --csv some_labeled_file.csv      # any labeled CSV with the model features

Run once after train.py. The model, preprocessing and threshold are loaded
as-is from artifacts/; nothing here is used to retune or reselect the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from malware_ml.inference import ProductionModel

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "splits" / "holdout_test.csv"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "holdout_results.json"))
    args = ap.parse_args()

    model = ProductionModel.load()
    df = pd.read_csv(args.csv, low_memory=False)
    result = model.evaluate_frame(df)
    result["evaluated_file"] = Path(args.csv).name
    Path(args.out).write_text(json.dumps(result, indent=2))

    m = result["metrics"]
    cm = result["confusion_matrix"]
    print(f"Model {result['model_name']} v{result['model_version']}  threshold={result['threshold']}")
    print(f"Samples evaluated: {result['n_evaluated']}  (invalid rows: {result['n_invalid']})")
    print(f"AUC:      {m['auc']}")
    print(f"Accuracy: {m['accuracy']}")
    print("Confusion matrix (rows = true label, columns = predicted label):")
    print(f"                    pred goodware(0)  pred malware(1)")
    print(f"  true goodware(0)  {cm['true_goodware_pred_goodware']:>16}  {cm['true_goodware_pred_malware']:>15}")
    print(f"  true malware(1)   {cm['true_malware_pred_goodware']:>16}  {cm['true_malware_pred_malware']:>15}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
