# 04_app_ui_keyed.py
# Streamlit UI with KEYED merge on 'cell_id' for TF-Exemplar; no positional assumptions.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
import numpy as np
from pathlib import Path
import warnings, sys

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
SCORED_H5AD    = Path(r"E:\tf_ZSCAPE_demo\ZSCAPE_full\zscape_tfap2a_scored.h5ad")
TFEX_ALIGNED_CSV = Path(r"E:\tf_ZSCAPE_demo\out_tfex\tfex_umap3d_for_ui.csv")  # from 03_verify_merge_keys.py

# --------------------------------------------------------------------------------------
# PAGE
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="ZSCAPE Perturbation Impact Explorer", page_icon="🎯", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 0.5rem; max-width: none; }
    .embedding-badge { background: linear-gradient(45deg, #ff6b6b, #4ecdc4); color: white; padding: 4px 12px; border-radius: 15px; font-size: 12px; font-weight: bold; margin-left: 10px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# LOADERS
# --------------------------------------------------------------------------------------
@st.cache_data
def load_scored() -> pd.DataFrame:
    st.write("Running load_scored()...")
    print("[UI] Loading scored AnnData:", SCORED_H5AD)
    if not SCORED_H5AD.exists():
        st.error(f"Missing scored H5AD: {SCORED_H5AD}")
        st.stop()

    adata = sc.read_h5ad(SCORED_H5AD)
    df = adata.obs.copy()
    df["cell_id"] = adata.obs_names.astype(str)

    # --- FIND PRECOMPUTED COORDS (prefer .obs columns to avoid recompute) ---
    used = None
    if all(c in df.columns for c in ("umap3d_1", "umap3d_2", "umap3d_3")):
        df["x"], df["y"], df["z"] = df["umap3d_1"], df["umap3d_2"], df["umap3d_3"]
        used = "obs.umap3d_*"
    elif all(c in df.columns for c in ("umap_1", "umap_2", "umap_3")):
        df["x"], df["y"], df["z"] = df["umap_1"], df["umap_2"], df["umap_3"]
        used = "obs.umap_* (3D)"
    elif "X_umap3d" in adata.obsm and adata.obsm["X_umap3d"].shape[1] == 3:
        um = adata.obsm["X_umap3d"]
        df["x"], df["y"], df["z"] = um[:, 0], um[:, 1], um[:, 2]
        used = "obsm['X_umap3d']"
    elif "X_umap" in adata.obsm and adata.obsm["X_umap"].shape[1] >= 2:
        um = adata.obsm["X_umap"]
        df["x"], df["y"] = um[:, 0], um[:, 1]
        df["z"] = um[:, 2] if um.shape[1] > 2 else np.random.randn(len(df)) * 0.05
        used = "obsm['X_umap'] (2D→fake z)"
    else:
        st.error(
            "No precomputed UMAP found in `.obs` (umap3d_* / umap_1..3) or `.obsm` "
            "('X_umap3d'/'X_umap'). To avoid a long compute, please add one of these to your H5AD."
        )
        st.stop()

    print(f"[UI] Using precomputed coordinates from: {used}")

    # Ensure expected columns exist
    for col in ["condition", "tissue", "cell_type_broad", "perturbation_z_score"]:
        if col not in df.columns:
            df[col] = "unknown"

    for col in ["condition", "tissue", "cell_type_broad"]:
        df[col] = df[col].astype("category")

    print(f"[UI] Scored frame: {len(df):,} rows | cols={list(df.columns)[:10]}...")
    return df


@st.cache_data
def load_tfex_aligned() -> pd.DataFrame | None:
    print("[UI] Loading TF-Ex aligned CSV:", TFEX_ALIGNED_CSV)
    if not TFEX_ALIGNED_CSV.exists():
        print("[UI] No TF-Ex aligned CSV found.")
        return None
    tf = pd.read_csv(TFEX_ALIGNED_CSV)
    # normalize columns
    need = {"cell_id", "umap_x", "umap_y", "umap_z"}
    if not need.issubset(tf.columns):
        st.warning(f"TF-Ex CSV missing columns: {sorted(list(need - set(tf.columns)))}")
        return None
    tf["cell_id"] = tf["cell_id"].astype(str)
    print(f"[UI] TF-Ex aligned: {len(tf):,} rows.")
    return tf

@st.cache_data
def build_tf_subset(main_df: pd.DataFrame, tf_aligned: pd.DataFrame | None):
    """Return (tf_df, stats) where tf_df has x_tfex/y_tfex/z_tfex and rows limited to TF cells."""
    if tf_aligned is None:
        return None, {"available": False, "n_tf": 0, "overlap": 0}

    # Left-join TF coords into main by cell_id
    tf = tf_aligned[["cell_id", "umap_x", "umap_y", "umap_z"]].copy()
    merged = main_df.merge(tf, on="cell_id", how="inner")  # keep only rows with TF coords
    merged = merged.rename(columns={"umap_x": "x_tfex", "umap_y": "y_tfex", "umap_z": "z_tfex"})
    n_tf = len(merged)
    stats = {"available": n_tf > 0, "n_tf": n_tf, "overlap": n_tf}
    print(f"[UI] TF subset built: {n_tf:,} rows.")
    return merged, stats

# --------------------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------------------
def make_3d(df, full_df, xcol, ycol, zcol, color_by, opacity=0.7, point_size=2.5,
            show_ghost=False, ghost_opacity=0.1, highlight_extreme=True, title=""):
    if df is None or df.empty:
        return None

    # sample for perf
    plot_df = df.sample(n=min(len(df), 25_000), random_state=42)

    fig = go.Figure()

    # Ghost layer first (same coordinate system as the active plot)
    if show_ghost and full_df is not None and not full_df.empty:
        ghost_df = full_df.sample(n=min(len(full_df), 15_000), random_state=42)
        fig.add_trace(go.Scatter3d(
            x=ghost_df[xcol], y=ghost_df[ycol], z=ghost_df[zcol],
            mode="markers",
            marker=dict(size=point_size*0.5, color="lightgray", opacity=ghost_opacity, line=dict(width=0)),
            name="All Cells (Ghost)",
            hoverinfo="skip",
            showlegend=(color_by != "perturbation_z_score"),
        ))

    if color_by == "perturbation_z_score" and "perturbation_z_score" in plot_df.columns:
        if highlight_extreme:
            size_arr = np.where(np.abs(plot_df["perturbation_z_score"]) > 2.0, point_size*1.5, point_size)
        else:
            size_arr = point_size
        fig.add_trace(go.Scatter3d(
            x=plot_df[xcol], y=plot_df[ycol], z=plot_df[zcol],
            mode="markers",
            marker=dict(
                size=size_arr,
                color=plot_df["perturbation_z_score"],
                colorscale="RdBu_r",
                cmin=-3, cmax=3,
                opacity=opacity, line=dict(width=0),
                colorbar=dict(title="Perturbation<br>Z-Score", x=1.02, len=0.8)
            ),
            text=[f"Condition: {r.get('condition','?')}<br>Tissue: {r.get('tissue','?')}<br>"
                  f"Cell Type: {r.get('cell_type_broad','?')}<br>Z: {r.get('perturbation_z_score',np.nan):.2f}"
                  for _, r in plot_df.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            showlegend=False,
            name="Cells",
        ))
    else:
        # categorical
        cats = [c for c in plot_df[color_by].unique() if pd.notna(c)]
        for c in cats:
            sub = plot_df[plot_df[color_by] == c]
            fig.add_trace(go.Scatter3d(
                x=sub[xcol], y=sub[ycol], z=sub[zcol],
                mode="markers",
                marker=dict(size=point_size, opacity=opacity, line=dict(width=0)),
                text=[f"Condition: {r.get('condition','?')}<br>Tissue: {r.get('tissue','?')}<br>"
                      f"Cell Type: {r.get('cell_type_broad','?')}<br>Z: {r.get('perturbation_z_score',np.nan):.2f}"
                      for _, r in sub.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name=str(c),
                showlegend=True
            ))

    fig.update_layout(
        scene=dict(
            bgcolor='#0e1117',
            xaxis=dict(backgroundcolor='#0e1117', gridcolor='rgba(255,255,255,0.1)', title=xcol),
            yaxis=dict(backgroundcolor='#0e1117', gridcolor='rgba(255,255,255,0.1)', title=ycol),
            zaxis=dict(backgroundcolor='#0e1117', gridcolor='rgba(255,255,255,0.1)', title=zcol)
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        height=1000,
        margin=dict(l=0, r=0, t=60, b=0),
        title=title,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(14,17,23,0.8)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1),
    )
    return fig

# --------------------------------------------------------------------------------------
# APP
# --------------------------------------------------------------------------------------
def main():
    st.markdown("<h1 class='main-title'>🎯 ZSCAPE Perturbation Impact Explorer</h1>", unsafe_allow_html=True)
    st.caption("Classical PCA/UMAP vs. TranscriptFormer (TF-Exemplar) — keyed by cell_id")

    # Load data
    df = load_scored()
    tf_aligned = load_tfex_aligned()
    tf_df, tf_stats = build_tf_subset(df, tf_aligned)

    # Sidebar
    st.sidebar.header("🎛️ Controls")

    tf_available = bool(tf_stats["available"])
    method = st.sidebar.selectbox(
        "Embedding",
        ["classical"] + (["tfex"] if tf_available else []),
        format_func=lambda x: "🔬 Classical PCA/UMAP" if x=="classical" else "🤖 TF-Exemplar (AI)"
    )

    # Status panel
    with st.sidebar.expander("Status", expanded=True):
        st.write(f"Scored cells: **{len(df):,}**")
        st.write(f"TF-Ex rows : **{tf_stats.get('n_tf', 0):,}** (joinable)")
        if not tf_available:
            st.info("TF-Ex embeddings not found or joinable. Using Classical only.")
        else:
            st.success("TF-Ex subset ready (keyed join).")

    color_by = st.sidebar.selectbox("Color Points By", ["perturbation_z_score", "condition", "tissue", "cell_type_broad"], index=0)
    point_size = st.sidebar.slider("Point Size", 0.5, 8.0, 2.5, 0.5)
    opacity = st.sidebar.slider("Point Opacity", 0.1, 1.0, 0.7, 0.05)
    highlight_extreme = st.sidebar.checkbox("Highlight Extreme Effects (|Z|>2)", True)
    show_ghost = st.sidebar.checkbox("Show Ghost Layer", False)
    ghost_opacity = st.sidebar.slider("Ghost Opacity", 0.01, 0.3, 0.1, 0.01) if show_ghost else 0.1

    # Filter block
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filter")
    filter_category = st.sidebar.selectbox("Filter Category", ["Population", "Tissue", "Cell Type"])
    if filter_category == "Population":
        opts = ["All"] + list(df["condition"].cat.categories)
        colname = "condition"
    elif filter_category == "Tissue":
        opts = ["All"] + list(df["tissue"].cat.categories)
        colname = "tissue"
    else:
        opts = ["All"] + list(df["cell_type_broad"].cat.categories)
        colname = "cell_type_broad"
    chosen = st.sidebar.selectbox("Select", opts, index=0)

    min_z, max_z = float(df["perturbation_z_score"].min()), float(df["perturbation_z_score"].max())
    z_range = st.sidebar.slider("Z-Score Range", min_z, max_z, (min_z, max_z))

    # Choose base table for active embedding (AND ghost must use the SAME coordinates)
    if method == "tfex":
        base = tf_df.copy()
        xcol, ycol, zcol = "x_tfex", "y_tfex", "z_tfex"
        badge = "🤖 AI-Powered"
        title = "TF-Exemplar Perturbation Landscape"
    else:
        base = df.copy()
        xcol, ycol, zcol = "x", "y", "z"
        badge = "🔬 Classical"
        title = "Classical UMAP Perturbation Landscape"

    # Apply filters on the active base
    if chosen != "All":
        base = base[base[colname] == chosen]
    base = base[base["perturbation_z_score"].between(*z_range)]

    if base.empty:
        st.warning("No cells match the selected filters.")
        return

    st.markdown(f"### 🌌 3D Impact Landscape ({len(base):,} cells) <span class='embedding-badge'>{badge}</span>", unsafe_allow_html=True)
    fig = make_3d(
        base,
        full_df=base if show_ghost else None,  # ghost layer uses the SAME coordinate space
        xcol=xcol, ycol=ycol, zcol=zcol,
        color_by=color_by, opacity=opacity, point_size=point_size,
        show_ghost=show_ghost, ghost_opacity=ghost_opacity,
        highlight_extreme=highlight_extreme,
        title=title
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    if color_by == "perturbation_z_score":
        st.markdown("""
**Interpretation**
- **Red**: strong positive response
- **Blue**: strong negative response  
- **Large points**: |Z| > 2 (if enabled)  
- In **TF-Exemplar** mode, positions reflect the foundation model’s semantic embedding; expect different geometry vs. classical UMAP.
""")

if __name__ == "__main__":
    # Helpful console prints for confidence
    print("================================================================================")
    print("Starting Streamlit app with keyed TF-Ex merge…")
    print("Python:", sys.version)
    print("================================================================================")
    main()
