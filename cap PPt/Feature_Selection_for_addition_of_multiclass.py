# ============================
# STAGE 2 — FEATURE SELECTION (robust, fallback for degenerate labels)
# FULL CODE (same logic) + ALSO SAVES y_multiclass alongside y_binary
#
# What changed vs your original:
# - Feature selection STILL uses ONLY y_binary (same as before)
# - y_multiclass is STILL excluded from features (no leakage)
# - When saving train/val/test_selected.parquet, we ALSO keep y_multiclass as an extra label column
# ============================

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG (edit)
# --------------------------
DATASET = "dataset23"                      # change to the dataset you want
DRIVE = "/content/drive/MyDrive"
SPLIT_DIR = f"{DRIVE}/IOT23_Split_FINAL/{DATASET}"

# Change this if you want a separate folder (recommended)
# OUT_DIR = f"{DRIVE}/IOT23_FeatureSelection_FINAL/{DATASET}"
OUT_DIR = f"{DRIVE}/IOT23_FeatureSelection_FINAL_WITH_VARIANT/{DATASET}"

os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_SIZE = 300_000      # increase if you want larger sample
TOP_K = 66                 # set to 10 if you want top-10 features
BATCH_ROWS = 50_000
RANDOM_STATE = 42

# Leak columns to exclude from features
LEAK_EXACT = {"ts", "y_multiclass", "id.orig_p", "id.resp_p"}
LEAK_PREFIX = ("conn_state_", "hist_")

def is_leak(c):
    return c in LEAK_EXACT or any(c.startswith(p) for p in LEAK_PREFIX)

# ----------------------------
# helper: streaming correlation scorer (memory-safe)
# ----------------------------
def streaming_pointbiserial_scores(parquet_path, exclude_cols=("y_binary",)):
    """
    Compute absolute Pearson correlation (point-biserial) between each numeric feature and y_binary
    in a streaming way (single pass, low memory).
    Returns pandas Series indexed by feature name (abs(correlation)).
    """
    pf = pq.ParquetFile(parquet_path)
    numeric_cols = None

    # accumulators
    n = 0
    sum_y = 0.0
    sum_y2 = 0.0
    sum_x = {}
    sum_x2 = {}
    sum_xy = {}

    for batch in pf.iter_batches(batch_size=BATCH_ROWS):
        df = batch.to_pandas()
        if "y_binary" not in df.columns:
            continue

        # identify numeric columns on first batch
        if numeric_cols is None:
            num = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in num if c not in exclude_cols and not is_leak(c)]
            for c in numeric_cols:
                sum_x[c] = 0.0
                sum_x2[c] = 0.0
                sum_xy[c] = 0.0

        y = pd.to_numeric(df["y_binary"], errors="coerce").fillna(0.0).astype(np.float64).to_numpy()
        m = len(y)
        if m == 0:
            continue

        sum_y += y.sum()
        sum_y2 += (y * y).sum()

        for c in numeric_cols:
            x = pd.to_numeric(df.get(c, pd.Series([0]*m)), errors="coerce").fillna(0.0).astype(np.float64).to_numpy()
            sum_x[c] += x.sum()
            sum_x2[c] += (x * x).sum()
            sum_xy[c] += (x * y).sum()

        n += m
        del df

    if numeric_cols is None or n == 0:
        raise RuntimeError("No numeric columns or no rows found in parquet file.")

    mean_y = sum_y / n
    var_y = sum_y2 / n - mean_y * mean_y

    scores = {}
    for c in numeric_cols:
        mean_x = sum_x[c] / n
        var_x = sum_x2[c] / n - mean_x * mean_x
        cov = sum_xy[c] / n - mean_x * mean_y
        denom = (var_x * var_y) ** 0.5 if var_x > 0 and var_y > 0 else 0.0
        corr = 0.0 if denom == 0.0 else cov / denom
        scores[c] = abs(float(corr))

    return pd.Series(scores).sort_values(ascending=False)

# ----------------------------
# 1) Build sample (streaming)
# ----------------------------
train_path = os.path.join(SPLIT_DIR, "train.parquet")
pf_train = pq.ParquetFile(train_path)

samples = []
rows_collected = 0
print("Building sample (streaming)...")

for batch in pf_train.iter_batches(batch_size=BATCH_ROWS):
    df = batch.to_pandas()

    # drop leak columns before sampling
    drop_cols = [c for c in df.columns if is_leak(c)]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df_num = df.select_dtypes(include=[np.number])
    if df_num.shape[0] == 0:
        continue

    take = min(len(df_num), SAMPLE_SIZE - rows_collected)
    if take <= 0:
        break

    samp = df_num.sample(n=take, random_state=RANDOM_STATE)
    samples.append(samp)
    rows_collected += take

    del df, df_num, samp

if not samples:
    raise RuntimeError("No numeric rows found for feature selection sample. Check dataset / leaks.")

sample_df = pd.concat(samples, ignore_index=True)
print(f"Collected sample rows: {len(sample_df)}")

if "y_binary" not in sample_df.columns:
    raise RuntimeError("y_binary missing from sample (can't train selector).")

y = pd.to_numeric(sample_df["y_binary"], errors="coerce").fillna(0).astype(int)
X = sample_df.drop(columns=["y_binary"], errors="ignore").select_dtypes(include=[np.number])

if X.shape[1] == 0:
    raise RuntimeError("No numeric features left after dropping leaks.")

# ----------------------------
# 2) Try XGBoost feature selection (if label distribution OK)
# ----------------------------
need_fallback = False
unique_labels = np.unique(y.values)
print("Label distribution in sample:", {int(k): int((y == k).sum()) for k in unique_labels})

if unique_labels.size < 2:
    print(" -> Only one class present in the sample. Will use streaming correlation fallback.")
    need_fallback = True

top_features = None
feature_importances = None

if not need_fallback:
    try:
        print("Training XGBoost for feature selection...")
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_va, label=y_va)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.05,
            "max_depth": 6,
            "seed": RANDOM_STATE,
            "verbosity": 1,
            "base_score": 0.5,
        }

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "eval")],
            early_stopping_rounds=30,
            verbose_eval=50,
        )

        score = bst.get_score(importance_type="gain")
        fi = pd.Series({c: score.get(f"f{i}", 0.0) for i, c in enumerate(X.columns)}).sort_values(ascending=False)

        feature_importances = fi
        top_features = fi.head(TOP_K).index.tolist()

        bst.save_model(os.path.join(OUT_DIR, "xgb_feature_selector.model"))
        fi.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"))
        print("XGBoost selection done.")

    except Exception as e:
        print("XGBoost failed (falling back):", str(e))
        need_fallback = True

# ----------------------------
# 3) Fallback: streaming correlation scoring
# ----------------------------
if need_fallback:
    print("Computing streaming point-biserial scores (fallback)...")
    scores = streaming_pointbiserial_scores(train_path)
    feature_importances = scores.sort_values(ascending=False)
    top_features = feature_importances.head(TOP_K).index.tolist()
    feature_importances.to_csv(os.path.join(OUT_DIR, "feature_importances_streaming.csv"))
    print("Fallback selection done.")

# write top features list
pd.Series(top_features).to_csv(os.path.join(OUT_DIR, "top_features.txt"), index=False)
print(f"Selected top {len(top_features)} features. Saved to {OUT_DIR}")

# ----------------------------
# 4) Save selected splits (streaming, robust)
#    IMPORTANT: Saves y_binary AND y_multiclass (if present), without using y_multiclass as a feature.
# ----------------------------
def save_selected(src_path, dst_path, top_features, batch_rows=BATCH_ROWS):
    pf = pq.ParquetFile(src_path)
    writer = None
    written = 0
    base_schema = None

    for batch in pf.iter_batches(batch_size=batch_rows):
        df = batch.to_pandas()

        # Must have y_binary to be usable for your binary model pipeline
        if "y_binary" not in df.columns:
            continue

        # keep: selected features + y_binary + (optionally) y_multiclass
        keep = [c for c in top_features if c in df.columns]
        keep.append("y_binary")
        if "y_multiclass" in df.columns:
            keep.append("y_multiclass")

        df_sel = df[keep].copy()

        # coerce types for safety
        for c in df_sel.columns:
            if c in ("y_binary", "y_multiclass"):
                df_sel.loc[:, c] = pd.to_numeric(df_sel[c], errors="coerce").fillna(0).astype("int32")
            else:
                df_sel.loc[:, c] = pd.to_numeric(df_sel[c], errors="coerce").fillna(0.0).astype("float32")

        table = pa.Table.from_pandas(df_sel, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(dst_path, table.schema, compression="snappy")
            base_schema = table.schema
        else:
            # ensure schema compatibility
            if not table.schema.equals(base_schema):
                cols = [f.name for f in base_schema]
                for c in cols:
                    if c not in df_sel.columns:
                        if c in ("y_binary", "y_multiclass"):
                            df_sel.loc[:, c] = 0
                        else:
                            df_sel.loc[:, c] = 0.0
                df_sel = df_sel[cols]
                table = pa.Table.from_pandas(df_sel, preserve_index=False)

        writer.write_table(table)
        written += len(df_sel)

        del df_sel, table, df

    if writer is not None:
        writer.close()

    return written

print("Writing selected feature splits (streaming)...")

train_in = os.path.join(SPLIT_DIR, "train.parquet")
val_in   = os.path.join(SPLIT_DIR, "val.parquet")
test_in  = os.path.join(SPLIT_DIR, "test.parquet")

train_out = os.path.join(OUT_DIR, "train_selected.parquet")
val_out   = os.path.join(OUT_DIR, "val_selected.parquet")
test_out  = os.path.join(OUT_DIR, "test_selected.parquet")

t_written = save_selected(train_in, train_out, top_features)
v_written = save_selected(val_in, val_out, top_features)
te_written = save_selected(test_in, test_out, top_features)

print("Saved rows:", {"train": t_written, "val": v_written, "test": te_written})
print("FEATURE SELECTION COMPLETE")
print("Output folder:", OUT_DIR)