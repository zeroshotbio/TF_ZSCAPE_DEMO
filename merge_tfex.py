# merge_tfex.py
import os, anndata as ad, numpy as np, pandas as pd

IN_DIR = r"E:\tf_ZSCAPE_demo\out_tfex"
OUT_H5 = r"E:\tf_ZSCAPE_demo\out_tfex\tfex_all.h5ad"

files = sorted([os.path.join(IN_DIR,f) for f in os.listdir(IN_DIR)
                if f.startswith("emb_tfap2a_ctrl_shard") and f.endswith(".h5ad")])
assert files, "No per-shard TF-Exemplar outputs found."

def pick_emb(a: ad.AnnData):
    # first try obsm keys that the CLI uses; else fall back to X
    keys = [k for k in a.obsm_keys() if "transcriptformer" in k.lower() or "emb" in k.lower()]
    return a.obsm[keys[0]] if keys else a.X

obs_all, emb_all = [], []
for f in files:
    a = ad.read_h5ad(f)
    E = np.asarray(pick_emb(a), dtype="float32")
    emb_all.append(E)
    obs_all.append(a.obs.copy())

E = np.vstack(emb_all)
obs = pd.concat(obs_all, axis=0)
A = ad.AnnData(X=E, obs=obs, dtype="float32")
A.obsm["X_tfex"] = A.X
A.write_h5ad(OUT_H5)
print("Wrote", OUT_H5, "cells:", A.n_obs, "emb_dim:", A.n_vars)
