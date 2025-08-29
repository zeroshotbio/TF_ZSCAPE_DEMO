import anndata as ad
import pandas as pd
import scanpy as sc
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def load_and_filter_tfap2a(data_path, use_precomputed_umap=True):
    """
    Load the ZSCAPE dataset and extract only populations perturbed with 'tfap2a'.
    """
    print("🧬 Loading ZSCAPE dataset...")
    
    # Load the main dataset
    if not Path(data_path).exists():
        raise FileNotFoundError(f"ZSCAPE data file not found: {data_path}")
    
    adata = ad.read_h5ad(data_path)
    print(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    # Explore gene_target values to identify tfap2a perturbations
    print(f"\n📊 Gene target distribution (Top 10):")
    target_counts = adata.obs['gene_target'].value_counts()
    print(target_counts.head(10))
    
    # Identify tfap2a perturbation populations
    tfap2a_conditions = [target for target in target_counts.index if 'tfap2a' in target]
    
    print(f"\n🎯 Identified 'tfap2a' conditions:")
    for condition in tfap2a_conditions:
        count = target_counts[condition]
        print(f"  • {condition}: {count:,} cells")
    
    # Filter to tfap2a cells only
    tfap2a_mask = adata.obs['gene_target'].isin(tfap2a_conditions)
    adata_tfap2a = adata[tfap2a_mask].copy()
    
    print(f"\n✅ Filtered to 'tfap2a' cells: {adata_tfap2a.shape[0]:,} cells")
    
    # Basic quality control stats
    print(f"\n📈 'tfap2a' cell quality metrics:")
    if 'total_counts' in adata_tfap2a.obs.columns:
        print(f"  • Mean UMIs per cell: {adata_tfap2a.obs['total_counts'].mean():.0f}")
        print(f"  • Mean genes per cell: {adata_tfap2a.obs['n_genes_by_counts'].mean():.0f}")
    
    # Check if pre-computed UMAP coordinates exist
    umap_cols = [col for col in adata_tfap2a.obs.columns if 'umap' in col.lower()]
    has_precomputed_umap = bool(umap_cols and use_precomputed_umap)
    
    if has_precomputed_umap:
        print(f"\n🗺️ Found pre-computed UMAP coordinates: {umap_cols}")
    else:
        print(f"\n🗺️ No pre-computed UMAP found, will generate new embeddings")
        
    return adata_tfap2a, has_precomputed_umap

def prepare_data(adata_subset, compute_new_embeddings=False):
    """
    Prepare data for visualization:
    1. Basic quality filtering
    2. Normalization
    3. Embedding generation (if needed)
    """
    print(f"\n🔧 Preparing data for visualization...")
    
    # Create a copy for processing
    adata = adata_subset.copy()
    
    # Basic quality control filtering
    print("Applying basic quality control...")
    
    gene_count_col = 'n_genes_by_counts'
    umi_count_col = 'total_counts'
    
    min_genes = 200  # Minimum genes per cell
    min_umis = 250   # Minimum UMIs per cell
    
    initial_cells = adata.shape[0]
    
    # Apply filters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if umi_count_col in adata.obs.columns:
        adata = adata[adata.obs[umi_count_col] >= min_umis]
    
    print(f"  • Filtered from {initial_cells:,} to {adata.shape[0]:,} cells")
    print(f"  • Criteria: ≥{min_genes} genes, ≥{min_umis} UMIs")
    
    # Normalize the data for downstream analysis
    print("Normalizing expression data...")
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes for embedding
    print("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    hvg_count = adata.var['highly_variable'].sum()
    print(f"  • Found {hvg_count} highly variable genes")
    
    # Generate new embeddings if requested or if no pre-computed ones exist
    if compute_new_embeddings:
        print("Computing new embeddings...")
        adata_hvg = adata[:, adata.var['highly_variable']]
        sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=50)
        sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)
        sc.tl.umap(adata_hvg, min_dist=0.1, spread=1.0)
        
        # Copy embeddings back to main object
        adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
        adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
        print(f"  • Generated UMAP embedding: {adata.obsm['X_umap'].shape}")
        
    return adata

def create_visualization_dataframe(adata, use_precomputed_umap=True):
    """
    Create a pandas DataFrame optimized for Streamlit visualization.
    """
    print(f"\n📊 Creating visualization dataframe...")
    
    # Extract coordinates for plotting
    if use_precomputed_umap and 'umap3d_1' in adata.obs.columns:
        print("Using pre-computed 3D UMAP coordinates")
        x_coords, y_coords = adata.obs['umap3d_1'].values, adata.obs['umap3d_2'].values
        z_coords = adata.obs['umap3d_3'].values if 'umap3d_3' in adata.obs.columns else None
        embedding_source = "precomputed_3d_umap"
    elif 'X_umap' in adata.obsm:
        print("Using computed 2D UMAP coordinates")
        x_coords, y_coords = adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1]
        z_coords = None
        embedding_source = "computed_2d_umap"
    else:
        raise ValueError("No embedding coordinates available!")

    # Create base dataframe
    viz_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'x': x_coords, 'y': y_coords,
        'gene_target': adata.obs['gene_target'].values,
    })
    
    if z_coords is not None:
        viz_df['z'] = z_coords

    # Add key metadata columns
    essential_columns = ['tissue', 'cell_type_broad', 'timepoint', 'total_counts', 'n_genes_by_counts', 'pct_counts_mt']
    for col in essential_columns:
        if col in adata.obs.columns:
            viz_df[col] = adata.obs[col].values

    # Rename columns for consistency
    viz_df = viz_df.rename(columns={
        'total_counts': 'n_umi', 'n_genes_by_counts': 'n_genes', 'pct_counts_mt': 'pct_mito'
    })
    
    # Create clean perturbation label
    def get_perturbation_type(target):
        if 'tfap2a' in target and '-' in target:
            return 'tfap2a Combo'
        elif 'tfap2a' == target:
            return 'tfap2a Single'
        return 'Other'
    viz_df['perturbation_type'] = viz_df['gene_target'].apply(get_perturbation_type)
    
    viz_df['embedding_source'] = embedding_source
    
    print(f"  • Created dataframe: {viz_df.shape}")
    print(f"  • Embedding source: {embedding_source}")
    print(f"  • Available columns: {list(viz_df.columns)}")
    
    # Summary of perturbation types
    perturbation_summary = viz_df['perturbation_type'].value_counts()
    print(f"  • Perturbation breakdown:")
    for p_type, count in perturbation_summary.items():
        print(f"    - {p_type}: {count:,} cells")
        
    return viz_df

def save_data(viz_df, output_dir="demo_data"):
    """Save prepared data for visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_file = output_path / "zscape_tfap2a_viz.csv"
    
    print(f"\n💾 Saving visualization data...")
    viz_df.to_csv(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  • Saved: {output_file}")
    print(f"  • File size: {file_size_mb:.1f} MB")
    
    return output_file

def main():
    """Main pipeline for preparing ZSCAPE tfap2a data."""
    print("🧬 ZSCAPE 'tfap2a' Perturbation - Data Preparation")
    print("=" * 60)
    
    # Path to your main ZSCAPE dataset
    data_path = "ZSCAPE_full/zscape_perturb_full_raw_counts.h5ad"
    
    try:
        # Step 1: Load and filter to tfap2a populations
        adata_tfap2a, has_precomputed_umap = load_and_filter_tfap2a(
            data_path, use_precomputed_umap=True
        )
        
        # Step 2: Prepare data (normalization, QC, embeddings)
        adata_processed = prepare_data(
            adata_tfap2a,
            compute_new_embeddings=not has_precomputed_umap
        )
        
        # Step 3: Create visualization dataframe
        viz_df = create_visualization_dataframe(
            adata_processed,
            use_precomputed_umap=has_precomputed_umap
        )
        
        # Step 4: Save for Streamlit visualization
        output_file = save_data(viz_df)
        
        print(f"\n✅ 'tfap2a' data preparation complete!")
        print(f"📁 Output: {output_file}")
        print(f"📊 Ready for visualization: {len(viz_df):,} cells")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()