# 01_map_symbols_to_ensembl.py
import anndata as ad
import pandas as pd
import json, re, os, sys
from datetime import datetime

# Optional fallback for unmapped symbols
try:
    from mygene import MyGeneInfo
    HAVE_MYG = True
except Exception:
    HAVE_MYG = False

# ----------- CONFIG -----------
H5AD_IN    = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_perturb_full_raw_counts.h5ad"
GENE_META  = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_perturb_full_gene_metadata.csv.gz"
VOCAB_JSON = r"E:\tf_ZSCAPE_demo\tf_exemplar\full_gene2idx.json"
H5AD_OUT   = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_perturb_full_raw_counts_ens.h5ad"
CSV_OUT    = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\symbol_to_ensembl_mapping.csv"
# --------------------------------

def strip_version(x):
    if x is None or pd.isna(x): 
        return None
    return re.sub(r"\.\d+$", "", str(x))

def looks_like_ensdarg(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.upper()
    frac = s.str.match(r"ENSDARG\d+(\.\d+)?").mean()
    return frac >= 0.80  # require strong evidence

def banner(msg):
    print("\n" + "="*88)
    print(msg)
    print("="*88)

def main():
    start = datetime.now()
    banner("🚀 START: Gene symbol → Ensembl ID mapping")

    print(f"[INFO] Input H5AD     : {H5AD_IN}")
    print(f"[INFO] Gene metadata  : {GENE_META}")
    print(f"[INFO] TF vocab JSON  : {VOCAB_JSON}")
    print(f"[INFO] Output H5AD    : {H5AD_OUT}")
    print(f"[INFO] Audit CSV      : {CSV_OUT}")
    print(f"[INFO] mygene fallback: {'ENABLED' if HAVE_MYG else 'DISABLED'}")

    if not os.path.exists(H5AD_IN):
        print(f"[ERROR] Missing file: {H5AD_IN}")
        sys.exit(1)
    if not os.path.exists(GENE_META):
        print(f"[ERROR] Missing file: {GENE_META}")
        sys.exit(1)
    if not os.path.exists(VOCAB_JSON):
        print(f"[ERROR] Missing file: {VOCAB_JSON}")
        sys.exit(1)

    banner("📥 Loading AnnData")
    adata = ad.read_h5ad(H5AD_IN)
    print(f"[OK] Loaded AnnData with shape X={adata.n_obs:,} cells × {adata.n_vars:,} genes")
    # Ensure unique var_names (anndata may add -1, -2…)
    before_names = adata.var_names.copy()
    adata.var_names_make_unique()
    made_unique = (before_names != adata.var_names).sum()
    if made_unique:
        print(f"[NOTE] Made var_names unique for {made_unique:,} entries")

    # Build base symbols by removing any anndata duplicate suffix "-<num>"
    base_symbols = pd.Series(adata.var_names, index=adata.var_names, dtype="string").str.replace(r"-\d+$", "", regex=True)
    n_changed = (base_symbols != pd.Series(adata.var_names, index=adata.var_names, dtype="string")).sum()
    print(f"[OK] Built base symbols (removed '-N' duplicates for {n_changed:,} genes)")
    dup_count = int(base_symbols.duplicated().sum())
    print(f"[INFO] Base symbol duplicates: {dup_count:,}")

    banner("📄 Loading ZSCAPE gene metadata")
    gmeta = pd.read_csv(GENE_META)
    print(f"[OK] Metadata rows: {len(gmeta):,}  Columns: {list(gmeta.columns)}")

    # Heuristics for column names
    symbol_col = None
    if "gene_short_name" in gmeta.columns:
        symbol_col = "gene_short_name"
    elif "symbol" in gmeta.columns:
        symbol_col = "symbol"

    ensembl_col = None
    if "id" in gmeta.columns and looks_like_ensdarg(gmeta["id"]):
        ensembl_col = "id"
    else:
        for c in gmeta.columns:
            if re.search("ensembl", c, flags=re.I):
                ensembl_col = c
                break

    if symbol_col is None or ensembl_col is None:
        print(f"[ERROR] Could not identify symbol/ensembl columns in metadata.\n"
              f"        Columns present: {list(gmeta.columns)}")
        sys.exit(1)

    print(f"[OK] Using symbol column    : '{symbol_col}'")
    print(f"[OK] Using Ensembl column   : '{ensembl_col}'")

    # Normalize & deduplicate metadata map
    m = gmeta[[symbol_col, ensembl_col]].dropna().copy()
    m[symbol_col] = m[symbol_col].astype(str)
    m[ensembl_col] = m[ensembl_col].map(strip_version)
    # Drop rows without a clean Ensembl after strip
    m = m[~m[ensembl_col].isna()]
    before_dups = len(m)
    m = m.drop_duplicates(subset=[symbol_col], keep="first")
    print(f"[OK] Metadata usable pairs  : {len(m):,} (dropped {before_dups - len(m):,} duplicate symbols)")
    # Primary mapping
    sym2ens = pd.Series(m[ensembl_col].values, index=m[symbol_col]).to_dict()

    banner("🧭 Applying metadata mapping to var symbols")
    mapped = base_symbols.map(sym2ens).astype("object")
    n_mapped = int(mapped.notna().sum())
    n_total  = int(len(mapped))
    print(f"[OK] Mapped by metadata     : {n_mapped:,} / {n_total:,} ({n_mapped/n_total:.2%})")

    # Optional mygene fallback
    n_left = int((~mapped.notna()).sum())
    if n_left > 0:
        print(f"[INFO] Unmapped after metadata: {n_left:,}")
        if HAVE_MYG:
            banner("🌐 mygene.info fallback")
            need_mask = mapped.isna()
            needed_syms = list(base_symbols[need_mask].unique())
            print(f"[INFO] Querying mygene for {len(needed_syms):,} unique symbols (zebrafish, 7955)")
            mg = MyGeneInfo()
            try:
                q = mg.querymany(
                    needed_syms,
                    scopes="symbol",
                    fields="ensembl.gene",
                    species=7955,
                    as_dataframe=True,
                    verbose=True,
                )
                if "ensembl.gene" in q.columns:
                    ens = q["ensembl.gene"].apply(lambda x: x[0] if isinstance(x, list) else x)
                    ens = ens.map(strip_version)
                    ens = ens[~ens.index.duplicated(keep="first")]
                    # Align to the base_symbols index needing fill
                    fill = ens.reindex(base_symbols[need_mask].values).set_axis(base_symbols[need_mask].index)
                    n_fill = int(fill.dropna().shape[0])
                    mapped.loc[fill.dropna().index] = fill.dropna().astype(str)
                    print(f"[OK] mygene filled          : {n_fill:,}")
                else:
                    print("[WARN] mygene returned no 'ensembl.gene' column")
            except Exception as e:
                print(f"[WARN] mygene query failed: {e}")
        else:
            print("[INFO] mygene fallback disabled; skipping.")

    # Attach to AnnData
    adata.var["gene_symbol_base"] = base_symbols.astype("object")
    adata.var["ensembl_id"] = mapped.astype("object")

    # Coverage summary
    coverage = float(mapped.notna().mean())
    n_unmapped = int((~mapped.notna()).sum())
    print(f"[SUMMARY] Ensembl coverage  : {coverage:.2%}   (mapped={n_total - n_unmapped:,}, unmapped={n_unmapped:,})")

    # Vocab overlap
    banner("📚 Checking overlap with TF-Exemplar vocab")
    with open(VOCAB_JSON, "r") as f:
        vocab_keys = json.load(f)
    vocab = set(map(str, vocab_keys.keys() if isinstance(vocab_keys, dict) else vocab_keys))
    mapped_ids = pd.Series([x for x in mapped.dropna().astype(str).unique()], dtype="object")
    overlap = int(mapped_ids.isin(list(vocab)).sum())
    overlap_pct = overlap / max(1, len(mapped_ids))
    print(f"[OK] Overlap with TF vocab  : {overlap:,} / {len(mapped_ids):,} ({overlap_pct:.2%})")

    # Top 10 unmapped preview
    if n_unmapped > 0:
        sample_unmapped = list(base_symbols[mapped.isna()].unique())[:10]
        print(f"[NOTE] Example unmapped base symbols (first 10): {sample_unmapped}")

    banner("🧾 Writing audit CSV")
    audit_df = pd.DataFrame(
        {
            "var_name": adata.var_names,
            "base_symbol": base_symbols,
            "ensembl_id": adata.var["ensembl_id"],
            "in_tf_vocab": [ (eid in vocab) if isinstance(eid, str) else False for eid in adata.var["ensembl_id"] ],
            "mapping_source": [
                "metadata" if (bs in sym2ens) else ("mygene" if (eid is not None and eid == adata.var['ensembl_id'][idx]) else "none")
                for idx, (bs, eid) in enumerate(zip(base_symbols, adata.var["ensembl_id"]))
            ],
        }
    )
    audit_df.to_csv(CSV_OUT, index=False)
    print(f"[OK] Wrote audit CSV → {CSV_OUT}  (rows={len(audit_df):,})")

    banner("💾 Saving H5AD with Ensembl IDs")
    adata.write_h5ad(H5AD_OUT)
    print(f"[OK] Saved AnnData → {H5AD_OUT}")

    elapsed = datetime.now() - start
    banner(f"✅ DONE in {elapsed}  |  Next step: TF-Exemplar UMAP (Script 2)")
    print()

if __name__ == "__main__":
    main()
