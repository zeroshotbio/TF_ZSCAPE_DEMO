import anndata as ad
import pandas as pd
import json, re

# Optional fallback for unmapped symbols
try:
    from mygene import MyGeneInfo
    HAVE_MYG = True
except Exception:
    HAVE_MYG = False

H5AD_IN   = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_perturb_full_raw_counts.h5ad"
GENE_META = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_perturb_full_gene_metadata.csv.gz"
VOCAB_JSON= r"E:\tf_ZSCAPE_demo\tf_exemplar\full_gene2idx.json"
H5AD_OUT  = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_perturb_full_raw_counts_ens.h5ad"
CSV_OUT   = r"E:\tf_ZSCAPE_demo\ZSCAPE_full\symbol_to_ensembl_mapping.csv"

def strip_version(x):
    if x is None or pd.isna(x): return None
    return re.sub(r"\.\d+$", "", str(x))

def looks_like_ensdarg(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.upper()
    frac = s.str.match(r"ENSDARG\d+(\.\d+)?").mean()
    return frac >= 0.80  # require strong evidence

print("→ Loading AnnData…")
adata = ad.read_h5ad(H5AD_IN)
adata.var_names_make_unique()

# Build "base" symbols by removing the numeric suffix that anndata added to resolve duplicates
base_symbols = pd.Series(adata.var_names, index=adata.var_names, dtype="string").str.replace(r"-\d+$", "", regex=True)

print("→ Loading ZSCAPE gene metadata…")
gmeta = pd.read_csv(GENE_META)

# Heuristics for column names used by ZSCAPE exports
symbol_col = None
if "gene_short_name" in gmeta.columns:
    symbol_col = "gene_short_name"
elif "symbol" in gmeta.columns:
    symbol_col = "symbol"

ensembl_col = None
if "id" in gmeta.columns and looks_like_ensdarg(gmeta["id"]):
    ensembl_col = "id"
else:
    # Fallback scan for any Ensembl-looking column
    for c in gmeta.columns:
        if re.search("ensembl", c, flags=re.I):
            ensembl_col = c
            break

if symbol_col is None or ensembl_col is None:
    raise RuntimeError(
        f"Could not identify symbol/ensembl columns in {GENE_META}. "
        f"Found: {list(gmeta.columns)}"
    )

print(f"→ Using symbol column '{symbol_col}' and Ensembl column '{ensembl_col}'.")

# Normalize and deduplicate the metadata map
m = gmeta[[symbol_col, ensembl_col]].dropna().copy()
m[symbol_col] = m[symbol_col].astype(str)
m[ensembl_col] = m[ensembl_col].map(strip_version)
m = m.drop_duplicates(subset=[symbol_col], keep="first")

# Primary mapping (ZSCAPE metadata): symbol -> Ensembl
sym2ens = pd.Series(m[ensembl_col].values, index=m[symbol_col]).to_dict()

# Apply map to "base" symbols (so duplicates like 'actb-1' still map via 'actb')
print("→ Applying metadata mapping…")
mapped = base_symbols.map(sym2ens)

# Optional fill with mygene for leftovers
need_mask = mapped.isna()
n_need = int(need_mask.sum())
if n_need > 0 and HAVE_MYG:
    print(f"→ Filling {n_need} unmapped symbols via mygene.info…")
    mg = MyGeneInfo()
    q = mg.querymany(
        list(base_symbols[need_mask].unique()),
        scopes="symbol",
        fields="ensembl.gene",
        species=7955,  # zebrafish
        as_dataframe=True,
        verbose=True,
    )
    ens = q["ensembl.gene"].apply(lambda x: x[0] if isinstance(x, list) else x)
    ens = ens.map(strip_version)
    # Ensure unique index for safe reindex
    ens = ens[~ens.index.duplicated(keep="first")]
    fill = ens.reindex(base_symbols[need_mask].values).set_axis(base_symbols[need_mask].index)
    mapped.loc[fill.dropna().index] = fill.dropna().astype(str)

# Attach to AnnData; force plain object dtype to placate h5ad writer
adata.var["ensembl_id"] = mapped.astype(object)

# Coverage and TF-Ex vocab overlap
coverage = float(mapped.notna().mean())
print(f"✓ Ensembl coverage from metadata+fallback: {coverage:.2%}")

with open(VOCAB_JSON, "r") as f:
    vocab = set(json.load(f).keys())

mapped_ids = pd.Series([x for x in mapped.dropna().astype(str).unique()], dtype="object")
overlap = int(mapped_ids.isin(list(vocab)).sum())
overlap_pct = overlap / max(1, len(mapped_ids))
print(f"✓ Overlap with TF-Exemplar vocab: {overlap} / {len(mapped_ids)} ({overlap_pct:.2%})")

# Audit CSV
pd.DataFrame(
    {
        "var_name": adata.var_names,
        "base_symbol": base_symbols,
        "ensembl_id": adata.var["ensembl_id"],
        "in_tf_vocab": [eid in vocab if isinstance(eid, str) else False for eid in adata.var["ensembl_id"]],
    }
).to_csv(CSV_OUT, index=False)
print(f"→ Wrote audit CSV: {CSV_OUT}")

# Save H5AD
adata.write_h5ad(H5AD_OUT)
print(f"→ Saved H5AD with Ensembl IDs: {H5AD_OUT}")
