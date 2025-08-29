# 02_tfex_umap3d.py
# Build a 3D UMAP from TF-Exemplar embeddings and export a UI-ready CSV
import os, sys
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from datetime import datetime

# ----------- CONFIG (edit if needed) -----------
IN_H5   = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_all.h5ad"
OUT_CSV = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_umap3d.csv"
OUT_H5  = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_all_with_umap3d.h5ad"
N_NEIGH = 30
MIN_DIST = 0.30
RANDOM_SEED = 42
# ----------------------------------------------

PREFERRED_KEYS = [
    "X_tfex",               # what we used in earlier sketches
    "X_transcriptformer",   # plausible
    "X_cell_emb",           # common name
    "X_emb",                # common name
    "X_embedding",          # generic
    "X_latent",             # generic
]
EXCLUDE_KEYS = {"X_umap", "X_umap3d", "X_pca"}

def banner(msg):
    print("\n" + "="*100)
    print(msg)
    print("="*100)

def pick_embedding_key(adata: ad.AnnData) -> str:
    # Prefer known names
    for k in PREFERRED_KEYS:
        if k in adata.obsm:
            X = adata.obsm[k]
            if isinstance(X, np.ndarray) and X.ndim == 2 and X.shape[1] >= 64:
                print(f"[OK] Using preferred embedding key '{k}' with shape {X.shape}")
                return k
    # Otherwise, auto-detect: any 2D float matrix with ≥64 dims that isn't excluded
    candidates = []
    for k, X in adata.obsm.items():
        if k in EXCLUDE_KEYS:
            continue
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype.kind in "fc" and X.shape[1] >= 64:
            candidates.append((k, X.shape[1], X.shape))
    if candidates:
        # Pick the widest (most dims)
        candidates.sort(key=lambda t: t[1], reverse=True)
        k = candidates[0][0]
        print(f"[OK] Auto-detected embedding key '{k}' with shape {candidates[0][2]}")
        return k
    return ""

def main():
    start = datetime.now()
    banner("🚀 START: Build 3D UMAP from TF-Exemplar embeddings")

    print(f"[INFO] Input H5AD : {IN_H5}")
    print(f"[INFO] Output CSV : {OUT_CSV}")
    print(f"[INFO] Output H5AD: {OUT_H5}")
    if not os.path.exists(IN_H5):
        print(f"[ERROR] Missing input: {IN_H5}")
        sys.exit(1)

    banner("📥 Loading AnnData")
    adata = ad.read_h5ad(IN_H5)
    print(f"[OK] Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    print(f"[INFO] obsm keys: {list(adata.obsm.keys())}")

    emb_key = pick_embedding_key(adata)
    if not emb_key:
        print("[ERROR] Could not find a TF-Exemplar embedding matrix in .obsm")
        print("        Expected one of:", PREFERRED_KEYS, "or a 2D float matrix with ≥64 dims.")
        sys.exit(1)

    banner("🧭 Computing neighbors on TF-Exemplar embedding")
    sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=N_NEIGH, metric="cosine")
    print(f"[OK] neighbors computed: use_rep='{emb_key}', n_neighbors={N_NEIGH}, metric='cosine'")

    banner("🌌 Computing 3D UMAP")
    sc.tl.umap(adata, n_components=3, min_dist=MIN_DIST, random_state=RANDOM_SEED)
    if "X_umap" not in adata.obsm or adata.obsm["X_umap"].shape[1] < 3:
        print("[ERROR] UMAP did not produce a 3D embedding.")
        sys.exit(1)
    um = adata.obsm["X_umap"]
    print(f"[OK] UMAP shape: {um.shape} (expect n_cells × 3)")

    banner("🧾 Building export dataframe")
    # Choose common annotation fields if present; fill with 'unknown' if missing
    wanted_obs = ["condition", "tissue", "cell_type_broad", "embryo", "gene_target"]
    present = [c for c in wanted_obs if c in adata.obs.columns]
    missing = [c for c in wanted_obs if c not in adata.obs.columns]
    if missing:
        print(f"[NOTE] Missing obs columns (filled as 'unknown'): {missing}")

    df = pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "umap_x": um[:, 0],
            "umap_y": um[:, 1],
            "umap_z": um[:, 2],
        }
    )
    for c in present:
        df[c] = adata.obs[c].astype(str).values
    for c in missing:
        df[c] = "unknown"

    # Basic sanity
    n_na = int(df[["umap_x", "umap_y", "umap_z"]].isna().any(axis=1).sum())
    if n_na:
        print(f"[WARN] {n_na:,} rows have NaNs in UMAP coords; these will remain in CSV for debugging.")

    # Write outputs
    banner("💾 Writing outputs")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Wrote CSV  → {OUT_CSV}  (rows={len(df):,})")

    # Also persist the AnnData with UMAP for later reuse
    adata.write_h5ad(OUT_H5)
    print(f"[OK] Wrote H5AD → {OUT_H5}")

    elapsed = datetime.now() - start
    banner(f"✅ DONE in {elapsed} | Next: wire UI merge on 'cell_id' (keyed join, not positional)")
    print()

if __name__ == "__main__":
    main()
