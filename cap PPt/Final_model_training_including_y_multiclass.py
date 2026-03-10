# ============================================================
# FINAL MULTICLASS (VARIANT) MODEL TRAINING — RAM SAFE (UPDATED)
# - Predicts y_multiclass (benign + variants)
# - Reads selected parquet that contains: features + y_binary + y_multiclass
# - NEVER loads full train/val/test into memory
# - Streams Parquet -> writes LIBSVM to LOCAL /content
# - Trains XGBoost multi:softprob
# - Same logic as before, but faster + safer against RAM/runtime issues
# ============================================================

import os
import json
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from datetime import datetime, timezone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss

# =======================
# CONFIG - EDIT THIS
# =======================
DRIVE_BASE = "/content/drive/MyDrive"
DATASET = "dataset23"  # <--- change per run

# Folder produced by your feature-selection script that SAVES y_multiclass
INPUT_DIR = os.path.join(DRIVE_BASE, "IOT23_FeatureSelection_FINAL_WITH_VARIANT", DATASET)

OUT_DIR = os.path.join(DRIVE_BASE, "IOT23_Models_Multiclass", DATASET)
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(INPUT_DIR, "train_selected.parquet")
VAL_FILE   = os.path.join(INPUT_DIR, "val_selected.parquet")
TEST_FILE  = os.path.join(INPUT_DIR, "test_selected.parquet")

# Safer batch size for huge datasets
PARQUET_BATCH = 10000

# IMPORTANT: write libsvm to LOCAL disk to avoid Drive lag + RAM spikes
LIBSVM_TEMP_DIR = f"/content/tmp_libsvm_multiclass_{DATASET}"
os.makedirs(LIBSVM_TEMP_DIR, exist_ok=True)

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 60

XGB_PARAMS_BASE = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 1,
    "seed": 42,
}

# =======================
# HELPERS
# =======================
def parquet_row_count(path):
    pf = pq.ParquetFile(path)
    return sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))

def read_schema_columns(path):
    pf = pq.ParquetFile(path)
    return list(pf.schema.names)

def stream_unique_labels(parquet_path, label_col="y_multiclass", batch_size=PARQUET_BATCH):
    pf = pq.ParquetFile(parquet_path)
    uniq = set()
    for batch in pf.iter_batches(batch_size=batch_size, columns=[label_col]):
        df = batch.to_pandas()
        if label_col not in df.columns:
            continue
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        uniq.update(np.unique(y).tolist())
        del df, y
    return sorted(list(uniq))

def max_index_in_libsvm(path, max_lines=None):
    mx = -1
    with open(path, "r") as f:
        for i, ln in enumerate(f):
            parts = ln.strip().split()
            for p in parts[1:]:
                if ":" in p:
                    try:
                        idx = int(p.split(":", 1)[0])
                        mx = max(mx, idx)
                    except Exception:
                        pass
            if max_lines is not None and i + 1 >= max_lines:
                break
    return mx

def parquet_to_libsvm_multiclass(parquet_path, out_path, feature_cols, label_col="y_multiclass", batch_size=PARQUET_BATCH):
    """
    Stream parquet -> LIBSVM file (RAM-safe).

    SAME LOGIC AS BEFORE:
    - 0-based indices
    - ensure each row has at least one idx:val
    - force last feature index to appear at least once overall

    UPDATED:
    - avoids slow df.iloc[i] row access
    - uses NumPy inside each batch
    - much safer for very large datasets
    """
    pf = pq.ParquetFile(parquet_path)
    last_index = len(feature_cols) - 1

    n_written = 0
    class_counts = {}
    first_row_written = False

    with open(out_path, "w") as w:
        for batch in pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()

            if label_col not in df.columns:
                raise RuntimeError(f"Label column '{label_col}' not found in {parquet_path}")

            # ensure feature cols exist + numeric
            for c in feature_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                else:
                    df[c] = 0.0

            y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
            X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

            for i in range(X.shape[0]):
                lbl = int(y[i])
                class_counts[lbl] = class_counts.get(lbl, 0) + 1

                row = X[i]
                nz_idx = np.flatnonzero(row)
                entries = [f"{j}:{row[j]:.6g}" for j in nz_idx]

                # ensure not empty line
                if len(entries) == 0 and last_index >= 0:
                    entries.append(f"{last_index}:0.0")

                # force last index to appear at least once in whole file
                if not first_row_written and last_index >= 0:
                    if last_index not in nz_idx:
                        entries.append(f"{last_index}:0.0")
                    first_row_written = True

                w.write(str(lbl) + " " + " ".join(entries) + "\n")
                n_written += 1

            del df, y, X

    return n_written, class_counts

def stream_true_labels(parquet_path, label_col="y_multiclass", batch_size=PARQUET_BATCH):
    pf = pq.ParquetFile(parquet_path)
    ys = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=[label_col]):
        df = batch.to_pandas()
        if label_col not in df.columns:
            continue
        ys.append(pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int))
        del df
    if not ys:
        return np.array([], dtype=int)
    return pd.concat(ys, ignore_index=True).to_numpy()

def evaluate_multiclass_probs(probs, y_true, num_class):
    """
    probs can be:
    - shape (N, K) for K>=2
    - shape (N,) if K==1 (degenerate)
    """
    out = {"n": int(len(y_true))}
    uniq, counts = np.unique(y_true, return_counts=True)
    out["class_counts"] = {int(k): int(v) for k, v in zip(uniq, counts)}

    # degenerate: only 1 class in training => probs may come 1D
    if num_class <= 1 or probs.ndim == 1:
        out["note"] = "degenerate_num_class_1"
        y_pred = np.zeros_like(y_true)
        out["accuracy"] = float(accuracy_score(y_true, y_pred)) if len(y_true) else None
        out["f1_macro"] = None
        out["f1_weighted"] = None
        out["logloss"] = 0.0
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist() if len(y_true) else None
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
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_class)))
        out["confusion_matrix"] = cm.tolist()
    else:
        out["confusion_matrix"] = "skipped (too many classes)"

    out["probs_mean"] = float(np.mean(probs)) if len(probs) else None
    return out

# =======================
# MAIN
# =======================
print("MULTICLASS TRAINING RUN (RAM-SAFE)")
print("DATASET:", DATASET)

for required in (TRAIN_FILE, VAL_FILE, TEST_FILE):
    if not os.path.exists(required):
        raise SystemExit(f"ERROR: missing required file: {required}")

cols = read_schema_columns(TRAIN_FILE)
if "y_multiclass" not in cols:
    raise SystemExit("ERROR: 'y_multiclass' not found in train_selected.parquet")

# features exclude BOTH labels
feature_cols = [c for c in cols if c not in ("y_binary", "y_multiclass")]
if len(feature_cols) == 0:
    raise SystemExit("ERROR: no feature columns found after excluding labels")

print(f"Detected {len(feature_cols)} features (excluding y_binary, y_multiclass).")

train_rows = parquet_row_count(TRAIN_FILE)
val_rows   = parquet_row_count(VAL_FILE)
test_rows  = parquet_row_count(TEST_FILE)
print(f"Row counts: train={train_rows:,} val={val_rows:,} test={test_rows:,}")

# discover classes from TRAIN only
train_classes = stream_unique_labels(TRAIN_FILE, label_col="y_multiclass")
num_class = (max(train_classes) + 1) if len(train_classes) else 1
print("Train classes:", train_classes[:40], ("..." if len(train_classes) > 40 else ""))
print("num_class:", num_class)

meta = {
    "dataset": DATASET,
    "task": "multiclass_variant_detection",
    "feature_count": int(len(feature_cols)),
    "train_rows": int(train_rows),
    "val_rows": int(val_rows),
    "test_rows": int(test_rows),
    "train_classes": [int(x) for x in train_classes],
    "num_class": int(num_class),
    "created_at": datetime.now(timezone.utc).isoformat(),
}

# Build params
XGB_PARAMS = dict(XGB_PARAMS_BASE)
XGB_PARAMS["num_class"] = int(num_class)

# Prepare LIBSVM paths (LOCAL)
train_libsvm = os.path.join(LIBSVM_TEMP_DIR, "train.svm")
val_libsvm   = os.path.join(LIBSVM_TEMP_DIR, "val.svm")
test_libsvm  = os.path.join(LIBSVM_TEMP_DIR, "test.svm")

try:
    print("Streaming parquet -> LIBSVM (LOCAL /content)...")
    n_train, ctrain = parquet_to_libsvm_multiclass(TRAIN_FILE, train_libsvm, feature_cols, label_col="y_multiclass")
    n_val,   cval   = parquet_to_libsvm_multiclass(VAL_FILE,   val_libsvm,   feature_cols, label_col="y_multiclass")
    n_test,  ctest  = parquet_to_libsvm_multiclass(TEST_FILE,  test_libsvm,  feature_cols, label_col="y_multiclass")

    meta["class_counts_train"] = {str(k): int(v) for k, v in ctrain.items()}
    meta["class_counts_val"]   = {str(k): int(v) for k, v in cval.items()}
    meta["class_counts_test"]  = {str(k): int(v) for k, v in ctest.items()}

    print("LIBSVM rows:", {"train": n_train, "val": n_val, "test": n_test})
    print("Train class_counts:", ctrain)

    if n_train == 0:
        meta["status"] = "no_train_rows"
        with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        raise SystemExit("ERROR: no train rows written.")

    # sanity: last feature index must appear
    mx = max_index_in_libsvm(train_libsvm, max_lines=200000)
    expected = len(feature_cols) - 1
    print("LIBSVM max index:", mx, "expected:", expected)
    if mx != expected:
        raise RuntimeError(f"LIBSVM index mismatch: got {mx}, expected {expected}")

    # Prepare DMatrices (external memory)
    dtrain = xgb.DMatrix(train_libsvm + "?format=libsvm", feature_names=feature_cols)
    dval   = xgb.DMatrix(val_libsvm   + "?format=libsvm", feature_names=feature_cols) if n_val > 0 else None
    dtest  = xgb.DMatrix(test_libsvm  + "?format=libsvm", feature_names=feature_cols) if n_test > 0 else None

    # If train has only one class, training is meaningless for variants; still save metadata + skip
    if len(ctrain.keys()) <= 1 or num_class <= 1:
        meta["status"] = "single_class_train"
        with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("WARNING: Only one class in training. Saved metadata and exiting.")
        raise SystemExit(0)

    # Train
    evals = [(dtrain, "train")]
    if dval is not None:
        evals.append((dval, "validation"))

    print("Starting XGBoost multiclass training...")
    bst = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50
    )

    model_path = os.path.join(OUT_DIR, "model_multiclass.xgb")
    bst.save_model(model_path)
    print("Saved model to:", model_path)

    # Save feature list
    feat_list_path = os.path.join(OUT_DIR, "feature_list_multiclass.txt")
    with open(feat_list_path, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    # EVALUATION (stream true labels; predict on dmat)
    metrics = {}

    def eval_split(dmat, parquet_path):
        if dmat is None:
            return None
        probs = bst.predict(dmat)
        y_true = stream_true_labels(parquet_path, label_col="y_multiclass")
        if y_true.size == 0:
            return None
        return evaluate_multiclass_probs(probs, y_true, num_class=num_class)

    print("Evaluating validation...")
    metrics["validation"] = eval_split(dval, VAL_FILE)

    print("Evaluating test...")
    metrics["test"] = eval_split(dtest, TEST_FILE)

    print("Evaluating train...")
    metrics["train"] = eval_split(dtrain, TRAIN_FILE)

    report = {"meta": meta, "metrics": metrics}
    with open(os.path.join(OUT_DIR, "training_report_multiclass.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("DONE ✅ Test summary:")
    print(json.dumps(metrics["test"], indent=2) if metrics.get("test") else "No test metrics")

finally:
    # cleanup LOCAL libsvm to free disk
    try:
        if os.path.exists(LIBSVM_TEMP_DIR):
            shutil.rmtree(LIBSVM_TEMP_DIR)
    except Exception as e:
        print("cleanup warning:", e)

print("Artifacts saved to:", OUT_DIR)