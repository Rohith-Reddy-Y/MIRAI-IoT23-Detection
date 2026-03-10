"""
Two-Stage Detection Server
Stage 1: Binary ensemble (BENIGN / MALICIOUS)
Stage 2: Multiclass variant detection (attack family identification)
Serves the dashboard + provides /api/predict endpoint.
"""
import os, sys, json, glob, traceback, time
import numpy as np
import pandas as pd
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Try multiple paths for binary models (nested vs flat)
_bm1 = os.path.join(BASE_DIR, "cap PPt", "IOT23_Models", "IOT23_Models")
_bm2 = os.path.join(BASE_DIR, "cap PPt", "IOT23_Models")
BINARY_MODELS_ROOT = _bm1 if os.path.exists(_bm1) else _bm2
MULTI_MODELS_ROOT = os.path.join(BASE_DIR, "cap PPt", "IOT23_Models_Multiclass")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB limit

# ── Selected models for detection (quality-filtered) ──
# Binary: Only models with ROC-AUC > 0.85 (exclude random/inverted models)
SELECTED_BINARY = {'dataset4','dataset5','dataset8','dataset9','dataset10','dataset17','dataset19','dataset20'}
# Multiclass: ALL trained models included to ensure every attack variant is covered
# Confidence-weighted voting ensures high-accuracy models dominate predictions
SELECTED_MULTI = {'dataset4','dataset5','dataset6','dataset7','dataset8','dataset9','dataset10','dataset11',
                  'dataset12','dataset13','dataset14','dataset15','dataset16','dataset17','dataset18',
                  'dataset19','dataset20','dataset21','dataset22','dataset23'}

# ── Per-dataset variant mappings (alphabetically sorted = LabelEncoder index) ──
# Each dataset has DIFFERENT detailed_label values
DATASET_VARIANTS = {
    "dataset1":  ["-"],
    "dataset2":  ["-"],
    "dataset3":  ["-"],
    "dataset4":  ["-", "PartOfAHorizontalPortScan"],
    "dataset5":  ["-", "PartOfAHorizontalPortScan"],
    "dataset6":  ["-", "C&C", "C&C-HeartBeat", "DDoS"],
    "dataset7":  ["-", "C&C-HeartBeat", "DDoS"],
    "dataset8":  ["-", "C&C", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"],
    "dataset9":  ["-", "C&C", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"],
    "dataset10": ["-", "Attack", "C&C", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"],
    "dataset11": ["-", "C&C"],
    "dataset12": ["-", "Attack", "C&C", "C&C-FileDownload", "C&C-HeartBeat", "DDoS", "PartOfAHorizontalPortScan"],
    "dataset13": ["-", "C&C-HeartBeat", "PartOfAHorizontalPortScan"],
    "dataset14": ["-", "C&C", "C&C-HeartBeat", "PartOfAHorizontalPortScan"],
    "dataset15": ["-", "C&C", "DDoS", "PartOfAHorizontalPortScan"],
    "dataset16": ["-", "C&C-Torii"],
    "dataset17": ["-", "C&C", "DDoS"],
    "dataset18": ["-", "C&C", "C&C-HeartBeat", "DDoS"],
    "dataset19": ["-", "C&C", "C&C-HeartBeat", "C&C-Mirai", "DDoS", "PartOfAHorizontalPortScan"],
    "dataset20": ["-", "Okiru"],
    "dataset21": ["-", "Okiru"],
    "dataset22": ["-", "Attack", "C&C-HeartBeat", "C&C-HeartBeat-Attack", "DDoS", "Okiru", "PartOfAHorizontalPortScan"],
    "dataset23": ["-", "C&C", "C&C-FileDownload", "C&C-HeartBeat", "DDoS", "FileDownload", "PartOfAHorizontalPortScan"],
}

VARIANT_DISPLAY = {
    "-": "Benign",
    "Attack": "Attack",
    "C&C": "C&C Communication",
    "C&C-FileDownload": "C&C File Download",
    "C&C-HeartBeat": "C&C Heartbeat",
    "C&C-HeartBeat-Attack": "C&C Heartbeat Attack",
    "C&C-HeartBeat-FileDownload": "C&C Heartbeat Download",
    "C&C-Mirai": "Mirai C&C",
    "C&C-Torii": "Torii C&C",
    "DDoS": "DDoS Attack",
    "FileDownload": "File Download",
    "Okiru": "Okiru Malware",
    "Okiru-Attack": "Okiru Attack",
    "PartOfAHorizontalPortScan": "Horizontal Port Scan",
    "PartOfAHorizontalPortScan-Attack": "Port Scan Attack"
}

# ── Load models at startup ──
BINARY_MODELS = []
MULTI_MODELS = []
LOAD_LOG = []

def load_binary_models():
    global BINARY_MODELS, LOAD_LOG
    LOAD_LOG.append(f"Binary root: {BINARY_MODELS_ROOT} exists={os.path.exists(BINARY_MODELS_ROOT)}")
    if not os.path.exists(BINARY_MODELS_ROOT):
        return
    try:
        import xgboost as xgb
        for ds_dir in sorted(glob.glob(os.path.join(BINARY_MODELS_ROOT, "dataset*"))):
            ds_name = os.path.basename(ds_dir)
            if ds_name not in SELECTED_BINARY:
                LOAD_LOG.append(f"Binary SKIP {ds_name} (not in selected)")
                continue
            mp = os.path.join(ds_dir, "model.xgb")
            fp = os.path.join(ds_dir, "feature_list.txt")
            if not os.path.exists(mp) or not os.path.exists(fp):
                continue
            try:
                with open(fp) as f:
                    feats = [ln.strip() for ln in f if ln.strip()]
                bst = xgb.Booster()
                bst.load_model(mp)
                BINARY_MODELS.append({"name": os.path.basename(ds_dir), "bst": bst, "features": feats})
            except Exception as e:
                LOAD_LOG.append(f"Binary ERR {os.path.basename(ds_dir)}: {e}")
    except Exception as e:
        LOAD_LOG.append(f"Binary FATAL: {e}")

def load_multiclass_models():
    global MULTI_MODELS, LOAD_LOG
    LOAD_LOG.append(f"Multi root: {MULTI_MODELS_ROOT} exists={os.path.exists(MULTI_MODELS_ROOT)}")
    if not os.path.exists(MULTI_MODELS_ROOT):
        return
    try:
        import xgboost as xgb
        for ds_dir in sorted(glob.glob(os.path.join(MULTI_MODELS_ROOT, "dataset*"))):
            ds_name = os.path.basename(ds_dir)
            if ds_name not in SELECTED_MULTI:
                LOAD_LOG.append(f"Multi SKIP {ds_name} (not in selected)")
                continue
            mp = os.path.join(ds_dir, "model_multiclass.xgb")
            fp = os.path.join(ds_dir, "feature_list.txt")
            rp = os.path.join(ds_dir, "training_report_multiclass.json")
            if not os.path.exists(mp) or not os.path.exists(fp):
                continue
            try:
                with open(fp) as f:
                    feats = [ln.strip() for ln in f if ln.strip()]
                bst = xgb.Booster()
                bst.load_model(mp)

                # Read training report for class count
                num_class = 2
                if os.path.exists(rp):
                    with open(rp) as f:
                        report = json.load(f)
                    num_class = report.get("meta", {}).get("num_class", 2)

                # Build CORRECT per-dataset variant map
                variant_map = {}
                ds_variants = DATASET_VARIANTS.get(ds_name, [])
                for idx in range(num_class):
                    if idx < len(ds_variants):
                        variant_map[idx] = ds_variants[idx]
                    else:
                        variant_map[idx] = f"Variant-{idx}"

                # Only load models with >1 class
                if num_class > 1:
                    MULTI_MODELS.append({
                        "name": ds_name,
                        "bst": bst,
                        "features": feats,
                        "num_class": num_class,
                        "variant_map": variant_map
                    })
                    LOAD_LOG.append(f"Multi OK {ds_name}: {num_class} classes → {[variant_map[i] for i in range(num_class)]}")
            except Exception as e:
                LOAD_LOG.append(f"Multi ERR {ds_name}: {e}")
    except Exception as e:
        LOAD_LOG.append(f"Multi FATAL: {e}")

try:
    load_binary_models()
    load_multiclass_models()
except Exception as e:
    LOAD_LOG.append(f"Load crash: {traceback.format_exc()}")

print(f"Loaded {len(BINARY_MODELS)} binary models, {len(MULTI_MODELS)} multiclass models", flush=True)

# ── Feature processing (same as user's code) ──
def one_hot_proto_service(df):
    out = df.copy()
    if "proto" in out.columns:
        p = out["proto"].astype(str).str.lower()
        out["proto_tcp"] = (p == "tcp").astype(float)
        out["proto_udp"] = (p == "udp").astype(float)
        out["proto_icmp"] = (p == "icmp").astype(float)
    if "service" in out.columns:
        s = out["service"].astype(str).str.lower()
        out["service_dns"] = (s == "dns").astype(float)
        out["service_http"] = (s == "http").astype(float)
        out["service_-"] = (s == "-").astype(float)
    return out

def prepare_features(df_raw, feature_list):
    df = df_raw.copy()
    df = df.drop(columns=["y_binary", "y_multiclass"], errors="ignore")
    df = one_hot_proto_service(df)
    for c in feature_list:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

# ── Stage 1: Binary Ensemble ──
def predict_binary_ensemble(df_raw, threshold=0.5, combine_rule="mean"):
    import xgboost as xgb
    per_probs = []
    for m in BINARY_MODELS:
        X = prepare_features(df_raw, m["features"])
        dmat = xgb.DMatrix(X, feature_names=m["features"])
        per_probs.append(m["bst"].predict(dmat))
    per_model_probs = np.vstack(per_probs).T
    final_prob = per_model_probs.mean(axis=1) if combine_rule == "mean" else per_model_probs.max(axis=1)
    final_pred = (final_prob >= threshold).astype(int)
    return final_prob, final_pred, per_model_probs

# ── Stage 2: Multiclass Variant Detection ──
def predict_variants(df_raw, malicious_mask):
    """Run ALL multiclass models on malicious rows. Best model auto-selected by confidence."""
    import xgboost as xgb
    n = len(df_raw)
    variant_names = ["N/A"] * n
    variant_confidence = [0.0] * n

    mal_indices = np.where(malicious_mask)[0]
    if len(mal_indices) == 0 or len(MULTI_MODELS) == 0:
        return variant_names, variant_confidence

    df_mal = df_raw.iloc[mal_indices].reset_index(drop=True)

    # For each malicious row, track the best (highest confidence) non-benign prediction
    best_variant = ["Unknown Malware"] * len(df_mal)
    best_conf = [0.0] * len(df_mal)
    best_model = [""] * len(df_mal)

    for m in MULTI_MODELS:
        try:
            X = prepare_features(df_mal, m["features"])
            dmat = xgb.DMatrix(X, feature_names=m["features"])
            probs = m["bst"].predict(dmat)

            if probs.ndim == 1:
                continue  # Skip single-output models

            # Reshape if needed (some models return flat array for 2-class)
            if probs.ndim == 2:
                pred_class = np.argmax(probs, axis=1)
                pred_conf = np.max(probs, axis=1)
            else:
                continue

            for i in range(len(df_mal)):
                cls = int(pred_class[i])
                conf = float(pred_conf[i])
                vname = m["variant_map"].get(cls, f"Variant-{cls}")

                # Skip benign predictions (we already know it's malicious from Stage 1)
                if vname == "-":
                    continue

                # Auto-select best: pick the prediction with highest confidence
                if conf > best_conf[i]:
                    best_conf[i] = conf
                    best_variant[i] = vname
                    best_model[i] = m["name"]
        except Exception as e:
            LOAD_LOG.append(f"Variant predict ERR {m['name']}: {e}")
            continue

    # Write results back to full-size arrays
    for i in range(len(df_mal)):
        orig_idx = mal_indices[i]
        display_name = VARIANT_DISPLAY.get(best_variant[i], best_variant[i])
        variant_names[orig_idx] = display_name
        variant_confidence[orig_idx] = round(best_conf[i], 4)

    return variant_names, variant_confidence

# ── Routes ──
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "binary_models": len(BINARY_MODELS),
        "multiclass_models": len(MULTI_MODELS),
        "log": LOAD_LOG[-20:]
    })

@app.route("/<path:filename>")
def static_files(filename):
    if filename.startswith("api/"):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(BASE_DIR, filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if len(BINARY_MODELS) == 0:
            return jsonify({"error": f"No binary models loaded. Check model path."}), 500
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        threshold = float(request.form.get("threshold", 0.5))
        combine = request.form.get("combine", "mean")

        if file.filename.lower().endswith(".parquet"):
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file)

        # Stage 1: Binary detection
        final_prob, final_pred, per_model_probs = predict_binary_ensemble(df, threshold, combine)
        model_names = [m["name"] for m in BINARY_MODELS]
        best_idx = per_model_probs.argmax(axis=1)

        # Stage 2: Variant detection for malicious packets
        variant_names, variant_confidence = predict_variants(df, final_pred == 1)

        # Build results
        results = []
        variant_counts = {}
        for i in range(min(len(df), 500)):
            vname = variant_names[i]
            entry = {
                "index": i,
                "probability": round(float(final_prob[i]), 6),
                "verdict": "MALICIOUS" if final_pred[i] == 1 else "BENIGN",
                "top_model": model_names[best_idx[i]],
                "top_model_conf": round(float(per_model_probs[i].max()), 6),
                "variant": vname,
                "variant_confidence": round(float(variant_confidence[i]), 4) if final_pred[i] == 1 else 0.0
            }
            results.append(entry)
            if final_pred[i] == 1 and vname != "N/A":
                variant_counts[vname] = variant_counts.get(vname, 0) + 1

        # Full variant counts (not just first 500)
        all_variant_counts = {}
        for i in range(len(df)):
            if final_pred[i] == 1:
                vname = variant_names[i]
                all_variant_counts[vname] = all_variant_counts.get(vname, 0) + 1

        # Per-model stats for charts
        model_stats = []
        for j, name in enumerate(model_names):
            probs = per_model_probs[:, j]
            flagged = int((probs >= threshold).sum())
            model_stats.append({"name": name, "flagged": flagged, "avg_prob": round(float(probs.mean()), 4)})

        n_mal = int(final_pred.sum())
        n_ben = int(len(df) - n_mal)
        return jsonify({
            "total": len(df),
            "benign": n_ben,
            "malicious": n_mal,
            "threat_pct": round(n_mal / len(df) * 100, 2) if len(df) > 0 else 0,
            "results": results,
            "model_stats": model_stats,
            "variant_counts": all_variant_counts,
            "multiclass_models_used": len(MULTI_MODELS),
            "prob_distribution": {
                "benign": [round(float(x), 4) for x in final_prob[final_pred == 0][:200]],
                "malicious": [round(float(x), 4) for x in final_prob[final_pred == 1][:200]]
            },
            "saved": False
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Dashboard: http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
