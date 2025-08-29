# tfex_umap3d.py
import scanpy as sc, pandas as pd, anndata as ad

IN_H5 = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_all.h5ad"
OUT_CSV = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_umap3d.csv"

adata = ad.read_h5ad(IN_H5)
sc.pp.neighbors(adata, use_rep="X_tfex", n_neighbors=30, metric="cosine")
sc.tl.umap(adata, n_components=3, min_dist=0.3)

um = adata.obsm["X_umap"]
df = pd.DataFrame({
    "cell_id": adata.obs_names,
    "umap_x": um[:,0],
    "umap_y": um[:,1],
    "umap_z": um[:,2],
    # keep your usual coloring fields:
    "condition": adata.obs.get("condition", "unknown"),
    "embryo": adata.obs.get("embryo", "unknown"),
    "gene_target": adata.obs.get("gene_target", "unknown"),
})
df.to_csv(OUT_CSV, index=False)
print("Wrote", OUT_CSV, "rows:", len(df))
