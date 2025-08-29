# 03_verify_merge_keys.py
# Verifies TF-Exemplar CSV <-> scored AnnData join keys and writes a UI-ready aligned CSV
import os, sys, json
import anndata as ad
import pandas as pd
from datetime import datetime

# -------- CONFIG --------
MAIN_H5   = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_tfap2a_scored.h5ad"   # your scored dataset used by the UI
TFEX_CSV  = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_umap3d.csv"                # from 02_tfex_umap3d.py
OUT_ALIGNED_CSV = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_umap3d_for_ui.csv"   # key-merge ready (filtered + deduped)
OUT_REPORT_JSON = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_merge_report.json"
# ------------------------

def banner(msg):
    print("\n" + "=" * 100)
    print(msg)
    print("=" * 100)

def main():
    start = datetime.now()
    banner("🚦 START: Verify merge keys (TF-Exemplar ↔ scored AnnData)")

    print(f"[INFO] Scored AnnData : {MAIN_H5}")
    print(f"[INFO] TF-Ex CSV      : {TFEX_CSV}")
    print(f"[INFO] Out aligned CSV: {OUT_ALIGNED_CSV}")

    if not os.path.exists(MAIN_H5):
        print(f"[ERROR] Missing scored AnnData: {MAIN_H5}")
        sys.exit(1)
    if not os.path.exists(TFEX_CSV):
        print(f"[ERROR] Missing TF-Ex CSV: {TFEX_CSV}")
        sys.exit(1)

    banner("📥 Load inputs")
    adata = ad.read_h5ad(MAIN_H5)
    print(f"[OK] AnnData loaded: {adata.n_obs:,} cells × {adata.n_vars:,} vars")
    main_ids = pd.Index(adata.obs_names.astype(str), name="cell_id")

    tfex = pd.read_csv(TFEX_CSV)
    print(f"[OK] TF-Ex CSV loaded: {len(tfex):,} rows, columns={list(tfex.columns)}")

    if "cell_id" not in tfex.columns:
        print("[ERROR] TF-Ex CSV is missing 'cell_id' column. Re-run 02_tfex_umap3d.py.")
        sys.exit(1)

    # Normalize key dtype
    tfex["cell_id"] = tfex["cell_id"].astype(str)

    banner("🔎 Basic key diagnostics")
    dup_tfex = tfex["cell_id"].duplicated().sum()
    print(f"[INFO] TF-Ex duplicate cell_id rows  : {dup_tfex:,}")

    n_tfex = len(tfex)
    inter = pd.Index(tfex["cell_id"]).intersection(main_ids)
    only_tfex = pd.Index(tfex["cell_id"]).difference(main_ids)
    only_main = main_ids.difference(pd.Index(tfex["cell_id"]))

    print(f"[INFO] In both (joinable)             : {len(inter):,} / {n_tfex:,} TF-Ex rows")
    print(f"[INFO] TF-Ex not in main              : {len(only_tfex):,}")
    print(f"[INFO] Main cells not in TF-Ex        : {len(only_main):,}")

    # Little peek at missing examples
    if len(only_tfex) > 0:
        print(f"[NOTE] Example TF-Ex IDs not in main  : {list(only_tfex[:5])}")
    if len(only_main) > 0:
        print(f"[NOTE] Example main IDs not in TF-Ex  : {list(only_main[:5])}")

    # Build aligned CSV (keep only rows that exist in main; drop duplicates, keep first)
    banner("🧭 Building aligned TF-Ex CSV for the UI (keyed by 'cell_id')")
    tfex_keep = tfex.drop_duplicates(subset=["cell_id"], keep="first")
    tfex_keep = tfex_keep[tfex_keep["cell_id"].isin(main_ids)]

    # Optional check of columns we expect for plotting/filters
    needed_cols = ["umap_x", "umap_y", "umap_z"]
    missing = [c for c in needed_cols if c not in tfex_keep.columns]
    if missing:
        print(f"[ERROR] TF-Ex CSV missing required columns: {missing}")
        sys.exit(1)

    # Reorder to main_ids order so a left-join preserves intuitive ordering if desired
    tfex_keep = tfex_keep.set_index("cell_id").loc[inter].reset_index()

    os.makedirs(os.path.dirname(OUT_ALIGNED_CSV), exist_ok=True)
    tfex_keep.to_csv(OUT_ALIGNED_CSV, index=False)
    print(f"[OK] Wrote aligned CSV → {OUT_ALIGNED_CSV} (rows={len(tfex_keep):,})")

    # Emit a structured report
    report = {
        "main_cells": int(adata.n_obs),
        "tfex_rows": int(n_tfex),
        "joinable": int(len(inter)),
        "tfex_only": int(len(only_tfex)),
        "main_only": int(len(only_main)),
        "tfex_duplicates": int(dup_tfex),
        "aligned_csv": OUT_ALIGNED_CSV,
    }
    with open(OUT_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Wrote report JSON → {OUT_REPORT_JSON}")
    print(json.dumps(report, indent=2))

    elapsed = datetime.now() - start
    banner(f"✅ DONE in {elapsed} | Next: update Streamlit to key-merge on 'cell_id' using the aligned CSV")

if __name__ == "__main__":
    main()
