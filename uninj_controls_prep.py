import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import umap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_filter_controls(data_path, use_precomputed_umap=True):
    """
    Load ZSCAPE dataset and extract only control populations:
    - Unperturbed cells 
    - ctrl-inj cells (injection controls)
    """
    print("🧬 Loading ZSCAPE dataset...")
    
    # Load the main dataset
    if not Path(data_path).exists():
        raise FileNotFoundError(f"ZSCAPE data file not found: {data_path}")
    
    adata = ad.read_h5ad(data_path)
    print(f"Loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    
    # Explore gene_target values to identify controls
    print(f"\n📊 Gene target distribution:")
    target_counts = adata.obs['gene_target'].value_counts()
    print(target_counts.head(10))
    
    # Identify control populations
    # Based on your summary: 34,686 unperturbed + 362,755 ctrl-inj
    control_conditions = []
    
    # Find unperturbed cells (might be labeled as 'unperturbed', 'control', 'wt', etc.)
    potential_unperturbed = [target for target in target_counts.index 
                           if any(keyword in target.lower() 
                                for keyword in ['unperturbed', 'wt', 'wildtype', 'control'])
                           and 'ctrl-inj' not in target.lower()]
    
    # Add ctrl-inj (injection control)
    ctrl_inj_targets = [target for target in target_counts.index if 'ctrl-inj' in target]
    
    control_conditions = potential_unperturbed + ctrl_inj_targets
    
    print(f"\n🎯 Identified control conditions:")
    for condition in control_conditions:
        count = target_counts[condition]
        print(f"  • {condition}: {count:,} cells")
    
    # Filter to control cells only
    control_mask = adata.obs['gene_target'].isin(control_conditions)
    adata_controls = adata[control_mask].copy()
    
    print(f"\n✅ Filtered to controls: {adata_controls.shape[0]:,} cells")
    
    # Basic quality control stats
    print(f"\n📈 Control cell quality metrics:")
    if 'total_counts' in adata_controls.obs.columns:
        print(f"  • Mean UMIs per cell: {adata_controls.obs['total_counts'].mean():.0f}")
        print(f"  • Mean genes per cell: {adata_controls.obs['n_genes_by_counts'].mean():.0f}")
    elif 'n.umi' in adata_controls.obs.columns:
        print(f"  • Mean UMIs per cell: {adata_controls.obs['n.umi'].mean():.0f}")
        print(f"  • Mean genes per cell: {adata_controls.obs['num_genes_expressed'].mean():.0f}")
    
    # Check if pre-computed UMAP coordinates exist
    umap_cols = [col for col in adata_controls.obs.columns if 'umap' in col.lower()]
    if umap_cols and use_precomputed_umap:
        print(f"\n🗺️ Found pre-computed UMAP coordinates: {umap_cols}")
        has_precomputed_umap = True
    else:
        print(f"\n🗺️ No pre-computed UMAP found, will generate new embeddings")
        has_precomputed_umap = False
    
    return adata_controls, has_precomputed_umap

def prepare_control_data(adata_controls, compute_new_embeddings=False):
    """
    Prepare control data for visualization:
    1. Basic quality filtering
    2. Normalization
    3. Embedding generation (if needed)
    """
    print(f"\n🔧 Preparing data for visualization...")
    
    # Create a copy for processing
    adata = adata_controls.copy()
    
    # Basic quality control filtering
    print("Applying basic quality control...")
    
    # Filter cells with very low gene counts (likely poor quality)
    if 'n_genes_by_counts' in adata.obs.columns:
        gene_count_col = 'n_genes_by_counts'
        umi_count_col = 'total_counts'
    else:
        gene_count_col = 'num_genes_expressed'
        umi_count_col = 'n.umi'
    
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
    
    # Store raw counts
    adata.raw = adata
    
    # Normalize to 10,000 reads per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # Log transform
    sc.pp.log1p(adata)
    
    # Find highly variable genes for embedding
    print("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    hvg_count = adata.var['highly_variable'].sum()
    print(f"  • Found {hvg_count} highly variable genes")
    
    # Generate new embeddings if requested or if no pre-computed ones exist
    if compute_new_embeddings:
        print("Computing new embeddings...")
        
        # Use highly variable genes for PCA
        adata_hvg = adata[:, adata.var['highly_variable']]
        
        # Principal component analysis
        sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=50)
        
        # Compute neighborhood graph
        sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=50)
        
        # UMAP embedding
        sc.tl.umap(adata_hvg, min_dist=0.1, spread=1.0)
        
        # Copy embeddings back to main object
        adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
        adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
        adata.obsp['distances'] = adata_hvg.obsp['distances']
        adata.obsp['connectivities'] = adata_hvg.obsp['connectivities']
        
        print(f"  • Generated PCA embedding: {adata.obsm['X_pca'].shape}")
        print(f"  • Generated UMAP embedding: {adata.obsm['X_umap'].shape}")
    
    return adata

def create_visualization_dataframe(adata, use_precomputed_umap=True):
    """
    Create a pandas DataFrame optimized for Streamlit visualization.
    """
    print(f"\n📊 Creating visualization dataframe...")
    
    # Extract coordinates for plotting
    if use_precomputed_umap and 'umap3d_1' in adata.obs.columns:
        # Use pre-computed 3D UMAP
        print("Using pre-computed 3D UMAP coordinates")
        x_coords = adata.obs['umap3d_1'].values
        y_coords = adata.obs['umap3d_2'].values
        z_coords = adata.obs['umap3d_3'].values if 'umap3d_3' in adata.obs.columns else None
        embedding_source = "precomputed_3d_umap"
        
    elif 'X_umap' in adata.obsm:
        # Use newly computed or existing 2D UMAP
        print("Using computed 2D UMAP coordinates")
        x_coords = adata.obsm['X_umap'][:, 0]
        y_coords = adata.obsm['X_umap'][:, 1]
        z_coords = None
        embedding_source = "computed_2d_umap"
        
    else:
        # Fallback to PCA if available
        if 'X_pca' in adata.obsm:
            print("Using PCA coordinates (PC1 vs PC2)")
            x_coords = adata.obsm['X_pca'][:, 0]
            y_coords = adata.obsm['X_pca'][:, 1]
            z_coords = adata.obsm['X_pca'][:, 2] if adata.obsm['X_pca'].shape[1] > 2 else None
            embedding_source = "pca"
        else:
            raise ValueError("No embedding coordinates available!")
    
    # Create base dataframe with essential columns
    viz_df = pd.DataFrame({
        'cell_id': adata.obs.index,
        'x': x_coords,
        'y': y_coords,
        'gene_target': adata.obs['gene_target'].values,
    })
    
    # Add z-coordinate if available
    if z_coords is not None:
        viz_df['z'] = z_coords
    
    # Add key metadata columns
    essential_columns = [
        'tissue', 'cell_type_broad', 'cell_type_sub', 'timepoint',
        'total_counts', 'n_genes_by_counts', 'pct_counts_mt',
        'n.umi', 'num_genes_expressed', 'perc_mitochondrial_umis'
    ]
    
    for col in essential_columns:
        if col in adata.obs.columns:
            viz_df[col] = adata.obs[col].values
    
    # Rename columns for consistency
    column_mapping = {
        'total_counts': 'n_umi',
        'n.umi': 'n_umi', 
        'n_genes_by_counts': 'n_genes',
        'num_genes_expressed': 'n_genes',
        'pct_counts_mt': 'pct_mito',
        'perc_mitochondrial_umis': 'pct_mito'
    }
    
    viz_df = viz_df.rename(columns=column_mapping)
    
    # Create clean control label
    viz_df['control_type'] = viz_df['gene_target'].apply(
        lambda x: 'Unperturbed' if 'ctrl-inj' not in x else 'Injection Control'
    )
    
    # Add embedding source info
    viz_df['embedding_source'] = embedding_source
    
    print(f"  • Created dataframe: {viz_df.shape}")
    print(f"  • Embedding source: {embedding_source}")
    print(f"  • Available columns: {list(viz_df.columns)}")
    
    # Summary of control types
    control_summary = viz_df['control_type'].value_counts()
    print(f"  • Control breakdown:")
    for control, count in control_summary.items():
        print(f"    - {control}: {count:,} cells")
    
    return viz_df

def save_controls_data(viz_df, output_dir="demo_data"):
    """Save prepared control data for visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    output_file = output_path / "zscape_controls_viz.csv"
    
    print(f"\n💾 Saving visualization data...")
    viz_df.to_csv(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  • Saved: {output_file}")
    print(f"  • File size: {file_size_mb:.1f} MB")
    
    return output_file

def main():
    """Main pipeline for preparing ZSCAPE control data."""
    print("🧬 ZSCAPE Control Populations - Data Preparation")
    print("=" * 60)
    print("Focus: Unperturbed + ctrl-inj control cells only")
    print("Target: ~397k cells (34,686 unperturbed + 362,755 ctrl-inj)")
    print()
    
    # Path to your main ZSCAPE dataset
    data_path = "ZSCAPE_full/zscape_perturb_full_raw_counts.h5ad"
    
    try:
        # Step 1: Load and filter to control populations
        adata_controls, has_precomputed_umap = load_and_filter_controls(
            data_path, 
            use_precomputed_umap=True
        )
        
        # Step 2: Prepare data (normalization, QC, embeddings)
        adata_processed = prepare_control_data(
            adata_controls,
            compute_new_embeddings=not has_precomputed_umap  # Only compute if no pre-computed
        )
        
        # Step 3: Create visualization dataframe
        viz_df = create_visualization_dataframe(
            adata_processed,
            use_precomputed_umap=has_precomputed_umap
        )
        
        # Step 4: Save for Streamlit visualization
        output_file = save_controls_data(viz_df)
        
        print(f"\n✅ Control data preparation complete!")
        print(f"📁 Output: {output_file}")
        print(f"📊 Ready for visualization: {len(viz_df):,} control cells")
        
        # Summary stats
        print(f"\n📈 Final dataset summary:")
        print(f"  • Total cells: {len(viz_df):,}")
        print(f"  • Control types: {viz_df['control_type'].nunique()}")
        print(f"  • Tissues: {viz_df['tissue'].nunique() if 'tissue' in viz_df.columns else 'N/A'}")
        print(f"  • Timepoints: {viz_df['timepoint'].nunique() if 'timepoint' in viz_df.columns else 'N/A'}")
        print(f"  • Embedding: {viz_df['embedding_source'].iloc[0]}")
        
        print(f"\n🚀 Next steps:")
        print(f"  1. Create Streamlit visualizer for control populations")
        print(f"  2. Explore tissue/timepoint distributions")
        print(f"  3. Use as baseline for perturbation comparisons")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()