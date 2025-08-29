import anndata as ad
import scanpy as sc
import pandas as pd
from pathlib import Path
import warnings
import time

warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path):
    """Load the full dataset and prepare subsets for analysis."""
    print("🧬 Loading and preparing ZSCAPE dataset...")
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    adata = ad.read_h5ad(data_path)
    print(f"  • Loaded full dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

    # --- Basic Filtering ---
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # --- FIX: Ensure gene names are unique ---
    adata.var_names_make_unique()
    
    print(f"  • Filtered dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes (unique gene names)")

    # --- Identify populations ---
    # Create a simple 'condition' column for DE analysis
    adata.obs['condition'] = 'other'
    control_mask = adata.obs['gene_target'].str.contains('ctrl-inj', na=False)
    tfap2a_mask = adata.obs['gene_target'].str.contains('tfap2a', na=False)
    
    adata.obs.loc[control_mask, 'condition'] = 'control'
    adata.obs.loc[tfap2a_mask, 'condition'] = 'tfap2a'

    # Create a combined object for analysis
    adata_analysis = adata[adata.obs['condition'].isin(['control', 'tfap2a'])].copy()
    print(f"  • Isolated for analysis: {adata_analysis.shape[0]:,} cells ('control' and 'tfap2a')")
    
    # Normalize and log-transform for DE and scoring
    sc.pp.normalize_total(adata_analysis, target_sum=1e4)
    sc.pp.log1p(adata_analysis)
    
    return adata_analysis

def run_celltype_specific_de(adata_analysis, n_top_genes=50):
    """
    Run DE analysis within each major cell type to find a robust signature.
    """
    print(f"\n🔬 Running cell type-specific differential expression...")
    start_time = time.time()
    
    all_signature_genes = set()
    
    # Find common cell types with enough cells in both conditions
    cell_types = adata_analysis.obs['cell_type_broad'].value_counts().index[:15] # Top 15 for efficiency
    
    for cell_type in cell_types:
        print(f"  • Analyzing: {cell_type}")
        
        # Subset data for the specific cell type
        adata_celltype = adata_analysis[adata_analysis.obs['cell_type_broad'] == cell_type]
        
        # Check if we have both conditions in this subset
        if len(adata_celltype.obs['condition'].unique()) < 2:
            print(f"    - Skipping, only one condition present.")
            continue
            
        # Perform DE analysis
        sc.tl.rank_genes_groups(
            adata_celltype, 
            groupby='condition', 
            groups=['tfap2a'], 
            reference='control',
            method='wilcoxon'
        )
        
        # Extract top up-regulated genes
        de_results = pd.DataFrame(adata_celltype.uns['rank_genes_groups']['names'])
        top_genes = de_results['tfap2a'].head(n_top_genes).tolist()
        all_signature_genes.update(top_genes)
        
    end_time = time.time()
    print(f"  • DE analysis complete in {end_time - start_time:.1f} seconds.")
    print(f"  • Compiled a master signature of {len(all_signature_genes)} unique genes.")
    
    return list(all_signature_genes)

def calculate_scores(adata_analysis, signature_genes):
    """Calculate raw and normalized Z-scores for perturbation."""
    print("\n🎯 Scoring all cells against the perturbation signature...")

    # --- Step 1: Calculate raw score using Scanpy's binned method ---
    sc.tl.score_genes(
        adata_analysis, 
        gene_list=signature_genes, 
        score_name='perturbation_raw_score',
        use_raw=False # Use the log-normalized data
    )
    print("  • Raw scores calculated.")

    # --- Step 2: Normalize to Z-score within each cell type ---
    # Calculate mean and std for ONLY control cells within each cell type
    control_stats = adata_analysis.obs[adata_analysis.obs['condition'] == 'control'].groupby('cell_type_broad')['perturbation_raw_score'].agg(['mean', 'std'])

    # **FIX**: Merge these stats directly into the main .obs dataframe
    adata_analysis.obs = adata_analysis.obs.merge(control_stats, on='cell_type_broad', how='left')

    # Calculate Z-score: (score - control_mean) / control_std
    # Add a small epsilon to std to avoid division by zero
    epsilon = 1e-9
    adata_analysis.obs['perturbation_z_score'] = (
        (adata_analysis.obs['perturbation_raw_score'] - adata_analysis.obs['mean']) / 
        (adata_analysis.obs['std'] + epsilon)
    )

    # Clean up temporary columns from the main .obs dataframe
    adata_analysis.obs.drop(columns=['mean', 'std'], inplace=True)

    print("  • Normalized Z-scores calculated.")

    # Display summary of scores
    score_summary = adata_analysis.obs.groupby('condition')['perturbation_z_score'].describe()
    print("\n📊 Z-Score Summary by Condition:")
    print(score_summary)

    return adata_analysis

def save_processed_data(adata_scored, output_file):
    """Save the final scored AnnData object."""
    print(f"\n💾 Saving final scored dataset...")
    
    # Ensure the parent directory exists
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    
    adata_scored.write_h5ad(output_file)
    print(f"  • Saved: {output_file}")

def main():
    """Main pipeline to score tfap2a perturbation effect."""
    print("🚀 Starting ZSCAPE Perturbation Scoring Pipeline")
    print("=" * 50)
    
    # Define file paths
    data_path = "ZSCAPE_full/zscape_perturb_full_raw_counts.h5ad"
    output_file = "ZSCAPE_full/zscape_tfap2a_scored.h5ad"
    
    try:
        # Step 1: Load and prepare data
        adata_analysis = load_and_prepare_data(data_path)
        
        # Step 2: Run DE to get the gene signature
        signature_genes = run_celltype_specific_de(adata_analysis)
        
        # Step 3: Calculate raw and Z-scores
        adata_scored = calculate_scores(adata_analysis, signature_genes)
        
        # Step 4: Save the final object
        save_processed_data(adata_scored, output_file)
        
        print(f"\n✅ Pipeline complete!")
        print(f"  • The new dataset with Z-scores is ready for visualization.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()