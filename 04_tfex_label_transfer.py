# 04_tfex_label_transfer.py
import os, json, time
import numpy as np
import pandas as pd
import anndata as ad

# Optional deps
try:
    import scanpy as sc
    HAVE_SCANPY = True
except Exception:
    HAVE_SCANPY = False

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import StratifiedKFold
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# ----------------- Paths & Config -----------------
IN_H5_CANDIDATES = [
    r"E:\tf_ZSCAPE_demo\out_tfex\tfex_all_with_umap3d.h5ad",
    r"E:\tf_ZSCAPE_demo\out_tfex\tfex_all.h5ad",
]
OUT_H5  = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_all_with_labels.h5ad"
OUT_CSV = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_labels_for_ui.csv"
REPORT  = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_label_report.json"

USE_REP               = "X_tfex"   # base representation
DO_PCA                = True       # denoise high-dim embedding a bit
N_PCS                 = 64
PCA_KEY               = "X_pca_tfex"
K_NEIGHBORS           = 25
LABELS                = ["tissue", "cell_type_broad"]
TRAIN_CONDITION       = "control"  # train on control labels only
MIN_CLASS_COUNT       = 25         # drop ultra-rare classes from training
CONFIDENCE_POWER      = 2.0        # weight = (1 - cosine_dist)^power
ABSTAIN_THRESHOLD     = 0.45       # if top prob < threshold -> "unknown"
CV_SPLITS             = 5          # quick sanity-check CV on control

# ----------------- Helpers -----------------
def ts():
    return time.strftime("%H:%M:%S")

def pick_input():
    for p in IN_H5_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No input H5AD found in candidates: {IN_H5_CANDIDATES}")

def l2_normalize(X, eps=1e-12):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return X / norm

def weighted_knn_predict(X_train, y_train, X_all, n_neighbors, power=2.0, abstain_thr=0.45):
    """Cosine kNN with weighted majority vote; returns preds, confs."""
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X_train)
    dist, idx = nn.kneighbors(X_all)  # dist in [0,2] for cosine
    sim = np.clip(1.0 - dist, 0.0, None)  # convert to similarity
    w = np.power(sim, power)
    classes, y_idx = np.unique(y_train, return_inverse=True)
    # Fast weighted voting:
    votes = np.zeros((X_all.shape[0], len(classes)), dtype=np.float64)
    for j in range(n_neighbors):
        cls_idx = y_idx[idx[:, j]]
        np.add.at(votes, (np.arange(X_all.shape[0]), cls_idx), w[:, j])
    top_idx = votes.argmax(1)
    top_w = votes[np.arange(votes.shape[0]), top_idx]
    sum_w = votes.sum(1) + 1e-12
    conf = top_w / sum_w
    preds = classes[top_idx].astype(object)
    preds[conf < abstain_thr] = "unknown"
    return preds, conf

def cv_score(X_train, y_train, n_splits=5):
    if len(np.unique(y_train)) < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, va in skf.split(X_train, y_train):
        p, _ = weighted_knn_predict(X_train[tr], y_train[tr], X_train[va], K_NEIGHBORS, CONFIDENCE_POWER, 0.0)
        accs.append((p == y_train[va]).mean())
    return float(np.mean(accs)), float(np.std(accs))

# ----------------- Main -----------------
print("\n" + "="*100)
print("🚀 START: TF-Exemplar label transfer (k-NN on embedding)")
print("="*100)
print(f"[{ts()}] INFO  scanpy    : {'OK' if HAVE_SCANPY else 'MISSING'}")
print(f"[{ts()}] INFO  scikit-learn: {'OK' if HAVE_SKLEARN else 'MISSING'}")
if not HAVE_SKLEARN:
    raise SystemExit("❌ scikit-learn not installed. Try: pip install scikit-learn")

IN_H5 = pick_input()
print("\n" + "="*100)
print("📥 Loading AnnData")
print("="*100)
print(f"[{ts()}] INFO  Input H5AD : {IN_H5}")
adata = ad.read_h5ad(IN_H5)
print(f"[{ts()}] OK    Loaded     : {adata.n_obs:,} cells × {adata.n_vars:,} vars")
print(f"[{ts()}] INFO  obsm keys  : {list(adata.obsm.keys())}")

if USE_REP not in adata.obsm:
    raise SystemExit(f"❌ Expected embedding obsm['{USE_REP}'] not found.")
X = adata.obsm[USE_REP]
print(f"[{ts()}] OK    Using rep  : '{USE_REP}' with shape {tuple(X.shape)}")

# Build/choose PCA
if DO_PCA:
    if PCA_KEY in adata.obsm and adata.obsm[PCA_KEY].shape[1] >= min(8, N_PCS):
        Xp = adata.obsm[PCA_KEY]
        print(f"[{ts()}] OK    Using cached '{PCA_KEY}' with shape {tuple(Xp.shape)}")
    else:
        if not HAVE_SCANPY:
            print(f"[{ts()}] WARN  scanpy not available; proceeding WITHOUT PCA denoising.")
            Xp = X
        else:
            print("\n" + "="*100)
            print("🧭 PCA denoising of TF-Exemplar embedding")
            print("="*100)
            # Scanpy PCA using the embedding as input
            tmp = ad.AnnData(X=X.copy())
            sc.pp.pca(tmp, n_comps=N_PCS, zero_center=True, svd_solver="arpack")
            Xp = tmp.obsm["X_pca"]
            adata.obsm[PCA_KEY] = Xp
            print(f"[{ts()}] OK    PCA done     : {tuple(Xp.shape)} → stored as obsm['{PCA_KEY}']")
else:
    Xp = X

# Normalize for cosine
X_all = l2_normalize(np.asarray(Xp, dtype=np.float32))

# Training mask (use control cells)
cond = adata.obs.get("condition", pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)).astype(str)
train_mask = cond.eq(TRAIN_CONDITION).values

summary = {
    "n_cells_total": int(adata.n_obs),
    "n_train_condition": TRAIN_CONDITION,
    "n_train_cells": int(train_mask.sum()),
    "embedding_key": PCA_KEY if DO_PCA else USE_REP,
    "k_neighbors": K_NEIGHBORS,
    "min_class_count": MIN_CLASS_COUNT,
    "abstain_threshold": ABSTAIN_THRESHOLD,
    "confidence_power": CONFIDENCE_POWER,
    "labels": {},
}

print("\n" + "="*100)
print(f"🎯 Training set (condition == '{TRAIN_CONDITION}')")
print("="*100)
print(f"[{ts()}] INFO  Train cells : {train_mask.sum():,} / {adata.n_obs:,}")

pred_cols = []
conf_cols = []

for label in LABELS:
    print("\n" + "-"*100)
    print(f"🏷️  Label: {label}")
    print("-"*100)
    y_full = adata.obs.get(label, pd.Series(["unknown"] * adata.n_obs, index=adata.obs_names)).astype(str).values
    y_train = y_full[train_mask]

    # Filter very rare classes from training
    vc = pd.Series(y_train).value_counts()
    keep_classes = vc[vc >= MIN_CLASS_COUNT].index.tolist()
    keep_mask = train_mask.copy()
    keep_mask[train_mask] = pd.Series(y_train).isin(keep_classes).values

    print(f"[{ts()}] INFO  Classes (>= {MIN_CLASS_COUNT} in train): {len(keep_classes)} / {vc.size}")
    print(f"[{ts()}] INFO  Train usable cells                   : {keep_mask.sum():,}")

    X_tr = X_all[keep_mask]
    y_tr = y_full[keep_mask]

    if len(np.unique(y_tr)) < 2:
        print(f"[{ts()}] WARN  Not enough classes for '{label}'. Skipping.")
        adata.obs[f"{label}_tfex_knn"] = "unknown"
        adata.obs[f"{label}_tfex_conf"] = 0.0
        summary["labels"][label] = {"train_cells": int(keep_mask.sum()), "classes": 1, "cv_acc_mean": None}
        continue

    # CV sanity check
    cv_mean, cv_std = cv_score(X_tr, y_tr, n_splits=CV_SPLITS)
    print(f"[{ts()}] OK    CV accuracy (control only, {CV_SPLITS} folds): {cv_mean:.3f} ± {cv_std:.3f}")

    # Fit and predict for all cells
    preds, conf = weighted_knn_predict(
        X_train=X_tr, y_train=y_tr, X_all=X_all,
        n_neighbors=K_NEIGHBORS, power=CONFIDENCE_POWER, abstain_thr=ABSTAIN_THRESHOLD
    )
    adata.obs[f"{label}_tfex_knn"] = preds
    adata.obs[f"{label}_tfex_conf"] = conf.astype(np.float32)
    pred_cols += [f"{label}_tfex_knn"]
    conf_cols += [f"{label}_tfex_conf"]

    # Little report
    known = preds != "unknown"
    print(f"[{ts()}] OK    Predicted for all cells. 'unknown' rate: {(~known).mean():.2%}")
    print(f"[{ts()}] INFO  Top 10 predicted {label}:")
    print(pd.Series(preds[known]).value_counts().head(10).to_string())

    summary["labels"][label] = {
        "train_cells": int(keep_mask.sum()),
        "n_classes_used": int(len(keep_classes)),
        "cv_acc_mean": float(cv_mean),
        "cv_acc_std": float(cv_std),
        "unknown_rate": float((~known).mean()),
    }

# ----------------- Outputs -----------------
print("\n" + "="*100)
print("💾 Writing outputs")
print("="*100)

# H5AD with labels
adata.write_h5ad(OUT_H5)
print(f"[{ts()}] OK    Wrote H5AD  → {OUT_H5}")

# UI CSV keyed by cell_id (obs_names)
ui = pd.DataFrame({"cell_id": adata.obs_names})
for c in pred_cols + conf_cols:
    ui[c] = adata.obs[c].values
ui.to_csv(OUT_CSV, index=False)
print(f"[{ts()}] OK    Wrote CSV   → {OUT_CSV}  (rows={len(ui):,})")

# JSON report
with open(REPORT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[{ts()}] OK    Wrote report→ {REPORT}")

print("\n" + "="*100)
print("✅ DONE | Next: load these *_tfex_knn columns in the UI when TF mode is active")
print("="*100)
