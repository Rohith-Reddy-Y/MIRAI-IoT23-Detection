# Batch evaluate all trained MULTICLASS XGBoost models and save per-model JSON + master summary
# Same logic/style as your binary evaluation script, but for y_multiclass

import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    log_loss
)
from datetime import datetime

# ========== CONFIG (edit these) ==========
MODELS_ROOT = "/content/drive/MyDrive/IOT23_Models_Multiclass"
FEATURE_SELECTION_ROOT = "/content/drive/MyDrive/IOT23_FeatureSelection_FINAL_WITH_VARIANT"
OUTPUT_ROOT = "/content/drive/MyDrive/IOT23_Evaluations_Multiclass"
BATCH_SIZE = 10000
# =========================================

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def evaluate_multiclass_probs(probs, y_true, num_class):
    """
    probs:
      - shape (N, K) for normal multiclass
      - shape (N,) if degenerate single-class case
    """
    out = {}
    out["n"] = int(len(y_true))

    uniq, counts = np.unique(y_true, return_counts=True)
    out["class_counts"] = {int(k): int(v) for k, v in zip(uniq, counts)}

    if len(y_true) == 0:
        out["note"] = "no_rows"
        return out

    # degenerate case
    if num_class <= 1 or probs.ndim == 1:
        out["note"] = "degenerate_num_class_1"
        y_pred = np.zeros_like(y_true)
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["f1_macro"] = None
        out["f1_weighted"] = None
        out["logloss"] = 0.0
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        out["probs_mean"] = float(np.mean(probs)) if len(probs) > 0 else None
        return out

    y_pred = np.argmax(probs, axis=1)

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))

    try:
        out["logloss"] = float(log_loss(y_true, probs, labels=list(range(num_class))))
    except Exception:
        out["logloss"] = None

    if num_class <= 30:
        out["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=list(range(num_class))
        ).tolist()
    else:
        out["confusion_matrix"] = "skipped (too many classes)"

    out["probs_mean"] = float(np.mean(probs)) if len(probs) > 0 else None
    return out


def stream_predict_and_evaluate_multiclass(model_path, test_parquet, feature_cols, batch_size=BATCH_SIZE):
    # load model
    bst = xgb.Booster()
    bst.load_model(model_path)

    pf = pq.ParquetFile(test_parquet)
    probs_parts = []
    y_parts = []

    for batch in pf.iter_batches(batch_size):
        df = batch.to_pandas()

        if "y_multiclass" not in df.columns:
            continue

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = 0.0

        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
        dmat = xgb.DMatrix(X, feature_names=feature_cols)
        probs = bst.predict(dmat)

        probs_parts.append(probs)
        y_parts.append(
            pd.to_numeric(df["y_multiclass"], errors="coerce").fillna(0).astype(int).to_numpy()
        )

        del df, X, dmat, probs

    if len(y_parts) == 0:
        raise RuntimeError("No labeled rows found in test_parquet.")

    # stack predictions
    if probs_parts[0].ndim == 1:
        probs_all = np.concatenate(probs_parts, axis=0)
        num_class = 1
    else:
        probs_all = np.vstack(probs_parts)
        num_class = probs_all.shape[1]

    y_all = np.concatenate(y_parts)

    return evaluate_multiclass_probs(probs_all, y_all, num_class=num_class)


# main loop: iterate over model folders
master_rows = []
datasets = sorted([d for d in os.listdir(MODELS_ROOT) if os.path.isdir(os.path.join(MODELS_ROOT, d))])

if not datasets:
    print("No dataset multiclass model folders found in", MODELS_ROOT)

for ds in datasets:
    print(f"\n--- Processing dataset: {ds}")
    ds_model_dir = os.path.join(MODELS_ROOT, ds)

    model_path = os.path.join(ds_model_dir, "model_multiclass.xgb")
    feat_path = os.path.join(ds_model_dir, "feature_list.txt")
    train_report_path = os.path.join(ds_model_dir, "training_report_multiclass.json")  # optional

    out_dir = os.path.join(OUTPUT_ROOT, ds)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "dataset": ds,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": model_path
    }

    # checks
    if not os.path.exists(model_path):
        report["error"] = f"Missing model file: {model_path}"
        json.dump(report, open(os.path.join(out_dir, "evaluation_report_multiclass.json"), "w"), indent=2)
        master_rows.append({"dataset": ds, "error": "missing_model"})
        print(" -> model missing, skipping.")
        continue

    if not os.path.exists(feat_path):
        report["error"] = f"Missing feature_list_multiclass.txt: {feat_path}"
        json.dump(report, open(os.path.join(out_dir, "evaluation_report_multiclass.json"), "w"), indent=2)
        master_rows.append({"dataset": ds, "error": "missing_feature_list"})
        print(" -> feature list missing, skipping.")
        continue

    # load feature list
    with open(feat_path, "r") as f:
        feature_cols = [line.strip() for line in f.readlines() if line.strip()]

    # find test parquet
    test_parquet = os.path.join(FEATURE_SELECTION_ROOT, ds, "test_selected.parquet")
    if not os.path.exists(test_parquet):
        report["error"] = f"Missing test_selected.parquet at {test_parquet}"
        json.dump(report, open(os.path.join(out_dir, "evaluation_report_multiclass.json"), "w"), indent=2)
        master_rows.append({"dataset": ds, "error": "missing_test_parquet"})
        print(" -> test parquet missing, skipping.")
        continue

    # run evaluation
    try:
        metrics = stream_predict_and_evaluate_multiclass(
            model_path=model_path,
            test_parquet=test_parquet,
            feature_cols=feature_cols,
            batch_size=BATCH_SIZE
        )

        report["test_metrics"] = metrics

        try:
            pf = pq.ParquetFile(test_parquet)
            rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
            report["test_rows"] = int(rows)
        except Exception:
            pass

        json.dump(report, open(os.path.join(out_dir, "evaluation_report_multiclass.json"), "w"), indent=2)
        print(" -> evaluation saved for", ds)

        master_row = {
            "dataset": ds,
            "test_rows": report.get("test_rows", None),
            "accuracy": metrics.get("accuracy", None),
            "f1_macro": metrics.get("f1_macro", None),
            "f1_weighted": metrics.get("f1_weighted", None),
            "logloss": metrics.get("logloss", None),
            "probs_mean": metrics.get("probs_mean", None)
        }
        master_rows.append(master_row)

    except Exception as e:
        report["error"] = f"evaluation_failure: {str(e)}"
        json.dump(report, open(os.path.join(out_dir, "evaluation_report_multiclass.json"), "w"), indent=2)
        master_rows.append({"dataset": ds, "error": "evaluation_failed", "msg": str(e)})
        print(" -> evaluation failed:", e)
        continue

# write master summary
if master_rows:
    df_master = pd.DataFrame(master_rows)
    master_csv = os.path.join(OUTPUT_ROOT, "master_summary_multiclass.csv")
    df_master.to_csv(master_csv, index=False)
    json.dump(master_rows, open(os.path.join(OUTPUT_ROOT, "master_summary_multiclass.json"), "w"), indent=2)
    print("\nMaster summary written to:", master_csv)
else:
    print("No results to write.")