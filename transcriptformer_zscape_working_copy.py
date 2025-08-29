import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scanpy as sc
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure Streamlit page
st.set_page_config(
    page_title="ZSCAPE Perturbation Impact Explorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS with minimal top padding
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main .block-container { 
        padding-left: 1rem; 
        padding-right: 1rem; 
        padding-top: 0.5rem;
        max-width: none; 
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; padding: 0 20px; background-color: #262730; border-radius: 5px; }
    .stTabs [aria-selected="true"] { background-color: #ff6b6b; }
    .metric-card { background-color: #262730; padding: 15px; border-radius: 8px; margin: 5px 0; }
    
    /* Main title styling with minimal top margin */
    .main-title {
        text-align: center; 
        color: #fafafa; 
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    
    .subtitle {
        text-align: center; 
        color: #aaaaaa; 
        font-size: 18px;
        margin-bottom: 20px;
    }
    
    .embedding-badge {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_scored_data():
    """Load the AnnData object with perturbation scores and prepare for visualization."""
    data_path = Path("ZSCAPE_full/zscape_tfap2a_scored.h5ad")

    if not data_path.exists():
        st.error(f"""
        Scored data not found: {data_path}
        
        Please run the scoring preparation script first:
        ```
        python tfap2a_scoring_prep.py
        ```
        """)
        st.stop()
    
    adata = sc.read_h5ad(data_path)
    
    # Create a pandas DataFrame for visualization
    viz_df = adata.obs.copy()
    
    # Add UMAP coordinates - try multiple possible column names
    coord_cols = ['umap3d_1', 'umap3d_2', 'umap3d_3', 'X_umap3d', 'umap_1', 'umap_2', 'umap_3']
    
    if 'umap3d_1' in viz_df.columns:
        viz_df['x'] = viz_df['umap3d_1']
        viz_df['y'] = viz_df['umap3d_2'] 
        viz_df['z'] = viz_df['umap3d_3']
    elif 'X_umap3d' in adata.obsm.keys():
        viz_df['x'] = adata.obsm['X_umap3d'][:, 0]
        viz_df['y'] = adata.obsm['X_umap3d'][:, 1]
        viz_df['z'] = adata.obsm['X_umap3d'][:, 2]
    else:
        st.error("Pre-computed 3D UMAP coordinates not found. Computing new embedding...")
        # Fallback: compute basic UMAP
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
        sc.tl.umap(adata, n_components=3)
        viz_df['x'] = adata.obsm['X_umap'][:, 0]
        viz_df['y'] = adata.obsm['X_umap'][:, 1]
        viz_df['z'] = adata.obsm['X_umap'][:, 2] if adata.obsm['X_umap'].shape[1] > 2 else np.random.randn(len(viz_df)) * 0.1
        
    return viz_df

@st.cache_data
def load_tfex_data():
    """Load TranscriptFormer embeddings if available."""
    tfex_csv_path = Path("out_tfex/tfex_umap3d.csv")
    
    if not tfex_csv_path.exists():
        return None
    
    try:
        tfex_df = pd.read_csv(tfex_csv_path)
        # Rename columns to match the main dataset
        tfex_df = tfex_df.rename(columns={
            'umap_x': 'x_tfex',
            'umap_y': 'y_tfex', 
            'umap_z': 'z_tfex'
        })
        return tfex_df
    except Exception as e:
        st.warning(f"Could not load TranscriptFormer embeddings: {e}")
        return None

@st.cache_data
def merge_datasets(main_df, tfex_df):
    """Merge main dataset with TranscriptFormer embeddings."""
    if tfex_df is None:
        return main_df, False
    
    try:
        # Filter main dataset to only control and tfap2a conditions (like TF-Exemplar input)
        filtered_main = main_df[main_df['condition'].isin(['control', 'tfap2a'])].copy()
        
        # Take first N cells to match TF-Exemplar output size
        if len(filtered_main) > len(tfex_df):
            filtered_main = filtered_main.head(len(tfex_df))
            st.sidebar.info(f"📊 TF-Exemplar mode: Showing {len(tfex_df):,} cells (control + tfap2a subset)")
        
        # Simple positional merge - assume same order
        if len(tfex_df) == len(filtered_main):
            filtered_main['x_tfex'] = tfex_df['umap_x'].values if 'umap_x' in tfex_df.columns else tfex_df['x_tfex'].values
            filtered_main['y_tfex'] = tfex_df['umap_y'].values if 'umap_y' in tfex_df.columns else tfex_df['y_tfex'].values  
            filtered_main['z_tfex'] = tfex_df['umap_z'].values if 'umap_z' in tfex_df.columns else tfex_df['z_tfex'].values
            merged_df = filtered_main
        else:
            st.warning(f"Size mismatch: filtered data has {len(filtered_main)} cells, TF-Exemplar has {len(tfex_df)} cells")
            return main_df, False
        
        # For classical mode, return full dataset; for TF mode, return filtered
        # We'll handle this in the main function
        tfex_available = not merged_df[['x_tfex', 'y_tfex', 'z_tfex']].isna().all().all()
        return main_df, tfex_available, merged_df  # Return both full and filtered
        
    except Exception as e:
        st.warning(f"Could not merge TranscriptFormer embeddings: {e}")
        return main_df, False, main_df

@st.cache_data
def calculate_impact_metrics(df):
    """Calculate tissue impact metrics based on Z-score distributions."""
    
    # Calculate median Z-score for tfap2a cells by tissue
    tfap2a_stats = df[df['condition'] == 'tfap2a'].groupby('tissue', observed=True)['perturbation_z_score'].agg([
        'median', 'mean', 'std', 'count'
    ]).round(3)
    
    # Calculate the "shift" - how much the median moves from 0 (control baseline)
    tfap2a_stats['impact_score'] = tfap2a_stats['median']
    
    # Calculate what percentage of cells in each tissue have strong perturbation (Z > 1.5)
    strong_perturbation = df[df['condition'] == 'tfap2a'].groupby('tissue', observed=True).apply(
        lambda x: (x['perturbation_z_score'] > 1.5).sum() / len(x) * 100, include_groups=False
    )
    
    # Ensure it's a Series
    if isinstance(strong_perturbation, pd.DataFrame):
        strong_perturbation = strong_perturbation.squeeze()
    
    tfap2a_stats['pct_strong_effect'] = strong_perturbation.round(1)
    
    # Sort by impact score (median Z-score)
    impact_ranking = tfap2a_stats.sort_values('impact_score', ascending=False)
    
    return impact_ranking

@st.cache_data
def filter_data(df, filter_category, filter_value, z_score_range, impact_threshold, enable_outlier_filter, buffer_sigma, max_points=25000):
    """Filter data based on user selections."""
    filtered_df = df.copy()

    # Apply single filter
    if filter_category == "Population" and filter_value != "All Populations":
        filtered_df = filtered_df[filtered_df['condition'] == filter_value]
    elif filter_category == "Tissue" and filter_value != "All Tissues":
        filtered_df = filtered_df[filtered_df['tissue'] == filter_value]
    elif filter_category == "Cell Type" and filter_value != "All Cell Types":
        filtered_df = filtered_df[filtered_df['cell_type_broad'] == filter_value]
    
    # Apply Z-score range filter
    if 'perturbation_z_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['perturbation_z_score'].between(*z_score_range)]
    
    # Filter by impact threshold - only show tissues with high impact
    if impact_threshold > 0:
        impact_metrics = calculate_impact_metrics(df)
        high_impact_tissues = impact_metrics[impact_metrics['impact_score'] >= impact_threshold].index.tolist()
        filtered_df = filtered_df[filtered_df['tissue'].isin(high_impact_tissues)]
    
    # Apply baseline subtraction filter (natural variation subtraction)
    if enable_outlier_filter:
        # Calculate the full range of natural variation for each cell type from control cells
        control_ranges = df[df['condition'] == 'control'].groupby('cell_type_broad')['perturbation_z_score'].agg(['min', 'max'])
        
        # Create masks for control and tfap2a cells
        tfap2a_mask = filtered_df['condition'] == 'tfap2a'
        control_mask = filtered_df['condition'] == 'control'
        
        # For tfap2a cells, only keep those OUTSIDE the natural variation range
        novel_state_mask = pd.Series(False, index=filtered_df.index)
        outlier_count = 0
        
        for cell_type in control_ranges.index:
            if cell_type in filtered_df['cell_type_broad'].values:
                cell_type_mask = (filtered_df['cell_type_broad'] == cell_type)
                tfap2a_cell_type_mask = tfap2a_mask & cell_type_mask
                
                # Get natural variation bounds with buffer
                min_natural = control_ranges.loc[cell_type, 'min'] - buffer_sigma
                max_natural = control_ranges.loc[cell_type, 'max'] + buffer_sigma
                
                # Mark tfap2a cells that create novel states (outside natural range)
                beyond_natural = (
                    tfap2a_cell_type_mask & 
                    ((filtered_df['perturbation_z_score'] > max_natural) | 
                     (filtered_df['perturbation_z_score'] < min_natural))
                )
                
                novel_state_mask |= beyond_natural
                outlier_count += beyond_natural.sum()
        
        # Keep all control cells + tfap2a cells with novel states
        filtered_df = filtered_df[control_mask | novel_state_mask]
        
        if outlier_count > 0:
            st.sidebar.success(f"🔬 Found {outlier_count:,} tfap2a cells with novel states (beyond natural range)")
        else:
            st.sidebar.info("🔍 No tfap2a cells found beyond natural variation range")
        
    # Sample for performance
    if len(filtered_df) > max_points:
        filtered_df = filtered_df.sample(n=max_points, random_state=42)
        st.sidebar.warning(f"⚠️ Showing random sample of {max_points:,} cells.")
        
    return filtered_df

def create_enhanced_3d_plot(df, full_df, embedding_method='classical', color_by='perturbation_z_score', opacity=0.7, point_size=2.5, highlight_extreme=True, show_ghost=False, ghost_opacity=0.1):
    """Create enhanced 3D plot highlighting perturbation effects."""
    if df.empty:
        return None

    plot_df = df.sample(n=min(len(df), 25000), random_state=42)
    sample_note = f"(showing {len(plot_df):,} of {len(df):,} cells)"

    # Select coordinates based on embedding method
    if embedding_method == 'tfex' and all(col in plot_df.columns for col in ['x_tfex', 'y_tfex', 'z_tfex']):
        x_col, y_col, z_col = 'x_tfex', 'y_tfex', 'z_tfex'
        method_label = "TF-Exemplar"
        title_prefix = "TF-Exemplar Perturbation Landscape"
    else:
        x_col, y_col, z_col = 'x', 'y', 'z'
        method_label = "Classical"
        title_prefix = "Classical UMAP Perturbation Landscape"
        if embedding_method == 'tfex':
            st.warning("⚠️ TranscriptFormer embeddings not available, using classical UMAP")

    fig = go.Figure()
    
    # Add ghost layer first (if enabled) - shows all cells in background
    if show_ghost:
        ghost_df = full_df.sample(n=min(len(full_df), 15000), random_state=42)
        if embedding_method == 'tfex' and all(col in ghost_df.columns for col in ['x_tfex', 'y_tfex', 'z_tfex']):
            ghost_x, ghost_y, ghost_z = ghost_df['x_tfex'], ghost_df['y_tfex'], ghost_df['z_tfex']
        else:
            ghost_x, ghost_y, ghost_z = ghost_df['x'], ghost_df['y'], ghost_df['z']
            
        fig.add_trace(go.Scatter3d(
            x=ghost_x,
            y=ghost_y, 
            z=ghost_z,
            mode='markers',
            marker=dict(
                size=point_size * 0.5,
                color='lightgray',
                opacity=ghost_opacity,
                line=dict(width=0)
            ),
            name='All Cells (Ghost)',
            hoverinfo='skip',
            showlegend=False  # Don't show in legend when using z-score coloring
        ))

    if color_by == 'perturbation_z_score':
        # Create a custom color scale that emphasizes extreme values
        plot_df['z_score_display'] = plot_df['perturbation_z_score'].copy()
        
        # Highlight cells with extreme perturbation effects
        if highlight_extreme:
            plot_df['point_size'] = np.where(
                abs(plot_df['perturbation_z_score']) > 2.0, point_size * 1.5, point_size
            )
            size_array = plot_df['point_size']
        else:
            size_array = point_size
            
        fig.add_trace(go.Scatter3d(
            x=plot_df[x_col],
            y=plot_df[y_col],
            z=plot_df[z_col],
            mode='markers',
            marker=dict(
                size=size_array,
                color=plot_df['z_score_display'],
                colorscale='RdBu_r',
                cmin=-3,
                cmax=3,
                opacity=opacity,
                line=dict(width=0),
                colorbar=dict(
                    title="Perturbation<br>Z-Score",
                    tickvals=[-3, -2, -1, 0, 1, 2, 3],
                    ticktext=["-3σ", "-2σ", "-1σ", "0", "+1σ", "+2σ", "+3σ"],
                    x=1.02,  # Position colorbar to avoid overlap
                    len=0.8   # Make colorbar shorter
                )
            ),
            text=[f"Condition: {row['condition']}<br>Tissue: {row['tissue']}<br>Cell Type: {row['cell_type_broad']}<br>Z-Score: {row['perturbation_z_score']:.2f}" 
                  for _, row in plot_df.iterrows()],
            hovertemplate='%{text}<extra></extra>',
            name='Filtered Cells',
            showlegend=False  # Don't show in legend to avoid overlap with colorbar
        ))
        
    else:
        # Standard categorical coloring - show ghost in legend for these
        if show_ghost:
            # Update ghost layer to show in legend for categorical plots
            fig.data[0].showlegend = True
        
        color_map = {'control': '#2E86AB', 'tfap2a': '#F24236'} if color_by == 'condition' else None
        
        if color_by == 'condition':
            for condition in plot_df[color_by].unique():
                condition_df = plot_df[plot_df[color_by] == condition]
                fig.add_trace(go.Scatter3d(
                    x=condition_df[x_col],
                    y=condition_df[y_col],
                    z=condition_df[z_col],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=color_map.get(condition, '#999999'),
                        opacity=opacity,
                        line=dict(width=0)
                    ),
                    text=[f"Condition: {row['condition']}<br>Tissue: {row['tissue']}<br>Cell Type: {row['cell_type_broad']}<br>Z-Score: {row['perturbation_z_score']:.2f}" 
                          for _, row in condition_df.iterrows()],
                    hovertemplate='%{text}<extra></extra>',
                    name=condition,
                    showlegend=True
                ))
        else:
            # For other categorical variables
            for category in plot_df[color_by].unique():
                if pd.notna(category):
                    cat_df = plot_df[plot_df[color_by] == category]
                    fig.add_trace(go.Scatter3d(
                        x=cat_df[x_col],
                        y=cat_df[y_col],
                        z=cat_df[z_col],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            opacity=opacity,
                            line=dict(width=0)
                        ),
                        text=[f"Condition: {row['condition']}<br>Tissue: {row['tissue']}<br>Cell Type: {row['cell_type_broad']}<br>Z-Score: {row['perturbation_z_score']:.2f}" 
                              for _, row in cat_df.iterrows()],
                        hovertemplate='%{text}<extra></extra>',
                        name=str(category),
                        showlegend=True
                    ))
    
    # Enhanced layout for better visibility
    fig.update_layout(
        scene=dict(
            bgcolor='#0e1117',
            xaxis=dict(backgroundcolor='#0e1117', gridcolor='rgba(255,255,255,0.1)', title=f'{method_label} Dim 1'),
            yaxis=dict(backgroundcolor='#0e1117', gridcolor='rgba(255,255,255,0.1)', title=f'{method_label} Dim 2'),
            zaxis=dict(backgroundcolor='#0e1117', gridcolor='rgba(255,255,255,0.1)', title=f'{method_label} Dim 3')
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        height=1400,
        margin=dict(l=0, r=0, t=60, b=0),
        title=f"{title_prefix} {sample_note}",
        legend=dict(
            x=0.02,  # Position legend on the left to avoid colorbar
            y=0.98,
            bgcolor='rgba(14, 17, 23, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        )
    )
    
    return fig

def create_impact_violin_plot(df, top_n=12):
    """Create split violin plots showing perturbation distribution by tissue."""
    
    # Calculate impact ranking
    impact_metrics = calculate_impact_metrics(df)
    top_tissues = impact_metrics.head(top_n).index.tolist()
    
    # Filter to top impacted tissues
    plot_df = df[df['tissue'].isin(top_tissues)].copy()
    
    # Create the plot
    fig = go.Figure()
    
    colors = {'control': '#2E86AB', 'tfap2a': '#F24236'}
    
    for i, tissue in enumerate(top_tissues):
        tissue_data = plot_df[plot_df['tissue'] == tissue]
        
        for j, condition in enumerate(['control', 'tfap2a']):
            condition_data = tissue_data[tissue_data['condition'] == condition]
            
            if len(condition_data) > 0:
                fig.add_trace(go.Violin(
                    y=condition_data['perturbation_z_score'],
                    x=[tissue] * len(condition_data),
                    name=f'{condition}',
                    side='negative' if condition == 'control' else 'positive',
                    line_color=colors[condition],
                    fillcolor=colors[condition],
                    opacity=0.7,
                    showlegend=(i == 0),  # Only show legend for first tissue
                    legendgroup=condition,
                    scalegroup=condition
                ))
    
    fig.update_layout(
        title="Perturbation Impact by Tissue (Split Violin Plot)",
        xaxis_title="Tissue (Ranked by Impact)",
        yaxis_title="Perturbation Z-Score",
        violinmode='overlay',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        height=600,
        xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.3)')
    )
    
    # Add horizontal reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Control Baseline")
    fig.add_hline(y=2, line_dash="dot", line_color="orange", opacity=0.5, annotation_text="Strong Effect (+2σ)")
    fig.add_hline(y=-2, line_dash="dot", line_color="orange", opacity=0.5)
    
    return fig

def display_impact_summary(df):
    """Display summary metrics of perturbation impact."""
    
    impact_metrics = calculate_impact_metrics(df)
    
    st.markdown("### 🎯 Tissue Impact Summary")
    
    # Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Most Impacted</h3>
            <h2>{impact_metrics.index[0]}</h2>
            <p>Z-score: {impact_metrics.iloc[0]['impact_score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        strong_effect_tissues = (impact_metrics['pct_strong_effect'] > 20).sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚡ High Impact Tissues</h3>
            <h2>{strong_effect_tissues}</h2>
            <p>>20% cells Z>1.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_strong = df[(df['condition'] == 'tfap2a') & (df['perturbation_z_score'] > 1.5)].shape[0]
        total_tfap2a = df[df['condition'] == 'tfap2a'].shape[0]
        pct_total = (total_strong / total_tfap2a * 100) if total_tfap2a > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌊 Overall Response</h3>
            <h2>{pct_total:.1f}%</h2>
            <p>cells strongly affected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        median_shift = df[df['condition'] == 'tfap2a']['perturbation_z_score'].median()
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Median Shift</h3>
            <h2>{median_shift:.2f}σ</h2>
            <p>from control baseline</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Impact ranking table
    st.markdown("#### 🏆 Tissue Impact Ranking")
    st.markdown("*Ranked by median Z-score of perturbed cells*")
    
    display_metrics = impact_metrics[['impact_score', 'pct_strong_effect', 'count']].copy()
    display_metrics.columns = ['Median Z-Score', '% Strong Effect (>1.5σ)', 'Cell Count']
    display_metrics = display_metrics.round(2)
    
    st.dataframe(display_metrics.head(10), use_container_width=True)

# --- Main App ---
def main():
    # Clear cache to avoid parameter mismatches and pandas issues
    st.cache_data.clear()
    
    # Main title with minimal padding
    st.markdown("""
    <h1 class='main-title'>🎯 ZSCAPE Perturbation Impact Explorer</h1>
    <p class='subtitle'>
        Visualizing tfap2a knockout effects • Classical analysis + TranscriptFormer AI embeddings
    </p>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading scored perturbation data..."):
        df = load_scored_data()
    
    # Load TranscriptFormer embeddings
    with st.spinner("Loading TranscriptFormer embeddings..."):
        tfex_df = load_tfex_data()
        df, tfex_available, tfex_subset_df = merge_datasets(df, tfex_df)
    
    # Get filter options (filtering out NaN values)
    populations = sorted([x for x in df['condition'].unique() if pd.notna(x)])
    tissues = sorted([x for x in df['tissue'].unique() if pd.notna(x)])
    cell_types = sorted([x for x in df['cell_type_broad'].unique() if pd.notna(x)])
    
    # --- Sidebar Controls ---
    st.sidebar.markdown("## 🎛️ Visualization Controls")

    # Embedding method selection
    if tfex_available:
        embedding_method = st.sidebar.selectbox(
            "🧠 Embedding Method",
            ['classical', 'tfex'],
            format_func=lambda x: "🔬 Classical PCA/UMAP" if x == 'classical' else "🤖 TF-Exemplar (AI)",
            help="Choose between classical dimensionality reduction or TranscriptFormer AI embeddings"
        )
        
        if embedding_method == 'tfex':
            st.sidebar.markdown("""
            <div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 8px; border-radius: 8px; margin: 5px 0;'>
                <b>🤖 AI-Powered Analysis</b><br>
                <small>Using TranscriptFormer foundation model embeddings</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        embedding_method = 'classical'
        st.sidebar.info("🔬 Using Classical PCA/UMAP embeddings")
        st.sidebar.markdown("*To enable TF-Exemplar AI embeddings, ensure `out_tfex/tfex_umap3d.csv` exists*")

    # Primary visualization options
    color_by = st.sidebar.selectbox(
        "Color Points By", 
        ['perturbation_z_score', 'condition', 'tissue', 'cell_type_broad'], 
        index=0,
        help="perturbation_z_score shows the intensity of the tfap2a effect"
    )
    
    highlight_extreme = st.sidebar.checkbox(
        "Highlight Extreme Effects", 
        value=True,
        help="Make cells with |Z-score| > 2 larger and more visible"
    )
    
    # Point size control
    point_size = st.sidebar.slider(
        "Point Size", 
        0.5, 8.0, 2.5, 0.5,
        help="Adjust the size of points in the 3D plot"
    )
    
    # Ghost layer control
    show_ghost = st.sidebar.checkbox(
        "Show Ghost Layer", 
        value=False,
        help="Show all cells as low-opacity background"
    )
    
    if show_ghost:
        ghost_opacity = st.sidebar.slider(
            "Ghost Layer Opacity", 
            0.01, 0.3, 0.1, 0.01,
            help="Adjust opacity of background ghost layer"
        )
    else:
        ghost_opacity = 0.1
    
    # Filtering controls - simplified to single selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filter")
    
    filter_category = st.sidebar.selectbox(
        "Filter Category",
        ["Population", "Tissue", "Cell Type"]
    )
    
    if filter_category == "Population":
        filter_options = ["All Populations"] + populations
    elif filter_category == "Tissue":
        filter_options = ["All Tissues"] + tissues
    else:  # Cell Type
        filter_options = ["All Cell Types"] + cell_types
    
    filter_value = st.sidebar.selectbox(
        f"Select {filter_category}",
        filter_options,
        index=0
    )
    
    # Z-score range filter
    min_z, max_z = float(df['perturbation_z_score'].min()), float(df['perturbation_z_score'].max())
    z_score_range = st.sidebar.slider(
        "Z-Score Range", 
        min_z, max_z, (min_z, max_z),
        help="Filter cells by their perturbation intensity"
    )
    
    # Tissue impact threshold filter
    impact_threshold = st.sidebar.slider(
        "Minimum Tissue Impact", 
        0.0, 2.0, 0.0, 0.1,
        help="Only show tissues with median Z-score above this threshold"
    )
    
    # Baseline subtraction filter
    enable_outlier_filter = st.sidebar.checkbox(
        "Baseline Subtraction Filter", 
        value=False,
        help="Show only tfap2a cells outside natural control variation range"
    )
    
    if enable_outlier_filter:
        buffer_sigma = st.sidebar.slider(
            "Detection Buffer", 
            0.0, 1.0, 0.1, 0.05,
            help="Additional buffer beyond natural range (in standard deviations)",
            format="%.2f σ"
        )
    else:
        buffer_sigma = 0.1

    st.sidebar.markdown("---")
    opacity = st.sidebar.slider("Point Opacity", 0.1, 1.0, 0.7, 0.05)

    # Data info
    st.sidebar.markdown("---")
    method_info = "🤖 TF-Exemplar AI" if embedding_method == 'tfex' and tfex_available else "🔬 Classical"
    st.sidebar.info(f"""
    **Dataset:** {len(df):,} cells total
    
    **Control:** {(df['condition'] == 'control').sum():,} cells
    
    **tfap2a:** {(df['condition'] == 'tfap2a').sum():,} cells
    
    **Z-Score Range:** {min_z:.2f} to {max_z:.2f}
    
    **Embedding:** {method_info}
    """)

    # --- Data Filtering ---
    # Use appropriate dataset based on embedding method
    if embedding_method == 'tfex' and tfex_available:
        base_df = tfex_subset_df  # Use the subset that has TF-Exemplar embeddings
    else:
        base_df = df  # Use full dataset for classical mode
    
    filtered_df = filter_data(base_df, filter_category, filter_value, z_score_range, impact_threshold, enable_outlier_filter, buffer_sigma)

    if filtered_df.empty:
        st.warning("No cells match the selected filters.")
        return

    # --- Main Visualization (primary focus) ---
    method_badge = "🤖 AI-Powered" if embedding_method == 'tfex' and tfex_available else "🔬 Classical"
    st.markdown(f"### 🌌 3D Impact Landscape ({len(filtered_df):,} cells) <span class='embedding-badge'>{method_badge}</span>", unsafe_allow_html=True)
    
    fig_3d = create_enhanced_3d_plot(
        filtered_df, 
        full_df=base_df,  # Use appropriate dataset for ghost layer
        embedding_method=embedding_method,
        color_by=color_by, 
        opacity=opacity,
        point_size=point_size,
        highlight_extreme=highlight_extreme,
        show_ghost=show_ghost,
        ghost_opacity=ghost_opacity
    )
    if fig_3d:
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Interpretation guide immediately below 3D viz
    if color_by == 'perturbation_z_score':
        interpretation_text = """
        **🎯 Interpretation Guide:**
        - **Red regions**: Cells with strong positive perturbation response (>normal variation)
        - **Blue regions**: Cells with strong negative response or unaffected  
        - **White/Gray**: Cells within normal variation range
        - **Large points**: Extreme effects (|Z-score| > 2σ) if highlighting enabled
        """
        
        if embedding_method == 'tfex' and tfex_available:
            interpretation_text += """
        - **🤖 AI Enhancement**: TranscriptFormer embeddings capture cross-species cell identity patterns
        - Foundation model may reveal perturbation effects invisible to classical dimensionality reduction
        """
        
        if enable_outlier_filter:
            interpretation_text += f"""
        - **🔬 Baseline Subtraction Active**: Natural control variation subtracted away (buffer: {buffer_sigma:.2f}σ)
        - Only showing tfap2a cells that create **entirely novel cellular states** beyond natural range
        - These represent genuine perturbation-induced changes, not natural variation amplification
        """
        
        st.markdown(interpretation_text)
    
    # --- Embedding Comparison Section ---
    if tfex_available and embedding_method in ['classical', 'tfex']:
        st.markdown("---")
        st.markdown("### 🔄 Embedding Method Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔬 Classical PCA/UMAP")
            st.markdown("""
            - **Technique**: Principal Component Analysis → UMAP projection
            - **Captures**: Linear gene expression patterns and local neighborhoods
            - **Strengths**: Fast, interpretable, well-established
            - **Limitations**: May miss complex gene regulatory relationships
            """)
        
        with col2:
            st.markdown("#### 🤖 TF-Exemplar AI Embeddings")
            st.markdown("""
            - **Technique**: Foundation model trained on millions of cells across species
            - **Captures**: Complex gene context, cell identity patterns, regulatory networks
            - **Strengths**: Cross-species knowledge, semantic understanding of cellular states
            - **Innovation**: May reveal perturbation effects invisible to classical methods
            """)
    
    # --- Impact Analysis (below main viz) ---
    st.markdown("---")
    
    # Impact summary
    display_impact_summary(df)  # Use full dataset for summary
    
    # Tissue impact distribution
    st.markdown("### 🎻 Tissue Impact Distribution")
    
    violin_fig = create_impact_violin_plot(df)  # Use full dataset
    st.plotly_chart(violin_fig, use_container_width=True)
    
    st.markdown("""
    **🎻 Violin Plot Guide:**
    - **Left side (Blue)**: Control cell distribution (should center on 0)
    - **Right side (Red)**: tfap2a perturbed cell distribution  
    - **Width**: Number of cells in that score range
    - **Shift**: How far the perturbation pushes cells from their natural baseline
    """)
    
    # --- Technical Details Section ---
    if tfex_available:
        with st.expander("🔧 Technical Details: TranscriptFormer Integration"):
            st.markdown("""
            ### How TF-Exemplar Embeddings Work
            
            **Foundation Model Training:**
            - Pre-trained on millions of single cells across multiple species including zebrafish
            - Learns semantic representations of cellular states and gene regulatory patterns
            - Captures cross-species conservation of cell identity programs
            
            **Integration Process:**
            1. **Data Preparation**: tfap2a vs control cells formatted as H5AD
            2. **Inference**: TF-Exemplar generates 512-dimensional embeddings per cell
            3. **Visualization**: 3D UMAP projection of AI embeddings
            4. **Analysis**: Same perturbation scoring applied to AI-organized cellular space
            
            **Expected Benefits:**
            - **Improved clustering**: AI may separate cell types more cleanly than PCA
            - **Enhanced signal**: Perturbation effects may be more pronounced in AI space  
            - **Novel insights**: Foundation model knowledge may reveal previously hidden patterns
            
            **Data Flow:**
            ```
            Raw Counts → TF-Exemplar → 512D Embeddings → UMAP 3D → Visualization
            ```
            """)

if __name__ == "__main__":
    main()