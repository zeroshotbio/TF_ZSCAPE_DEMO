import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scanpy as sc
from pathlib import Path
import numpy as np
import json
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
    
    .embedding-badge-classical {
        background: #4a5568;
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 10px;
        border: 1px solid #6b7280;
    }
    
    .embedding-badge-ai {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 10px;
        box-shadow: 0 2px 4px rgba(255, 107, 107, 0.3);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_scored_data():
    """Load the AnnData object with perturbation scores and prepare for visualization."""
    data_path = Path("ZSCAPE_full/zscape_tfap2a_scored.h5ad")

    if not data_path.exists():
        st.error(f"Scored data not found: {data_path}")
        st.stop()
    
    adata = sc.read_h5ad(data_path)
    
    # Create a pandas DataFrame for visualization
    viz_df = adata.obs.copy()
    
    # CRITICAL: Set cell_id from index (this is what the TF outputs use as key)
    viz_df['cell_id'] = viz_df.index.astype(str)
    
    # Add UMAP coordinates - try multiple possible locations
    if 'umap3d_1' in viz_df.columns:
        viz_df['x'] = viz_df['umap3d_1']
        viz_df['y'] = viz_df['umap3d_2'] 
        viz_df['z'] = viz_df['umap3d_3']
    elif 'X_umap3d' in adata.obsm.keys():
        viz_df['x'] = adata.obsm['X_umap3d'][:, 0]
        viz_df['y'] = adata.obsm['X_umap3d'][:, 1]
        viz_df['z'] = adata.obsm['X_umap3d'][:, 2]
    else:
        st.warning("Pre-computed 3D UMAP not found. Computing new embedding...")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
        sc.tl.umap(adata, n_components=3)
        viz_df['x'] = adata.obsm['X_umap'][:, 0]
        viz_df['y'] = adata.obsm['X_umap'][:, 1]
        viz_df['z'] = adata.obsm['X_umap'][:, 2] if adata.obsm['X_umap'].shape[1] > 2 else np.random.randn(len(viz_df)) * 0.1
        
    return viz_df

@st.cache_data
def load_tfex_data():
    """Load TranscriptFormer UMAP coordinates."""
    # Try both possible file names
    tfex_csv_paths = [
        Path("out_tfex/tfex_umap3d_for_ui.csv"),
        Path("out_tfex/tfex_umap3d.csv")
    ]
    
    for path in tfex_csv_paths:
        if path.exists():
            try:
                tfex_df = pd.read_csv(path)
                # Ensure cell_id is string
                tfex_df['cell_id'] = tfex_df['cell_id'].astype(str)
                return tfex_df
            except Exception as e:
                st.warning(f"Could not load TF UMAP from {path}: {e}")
                continue
    
    return None

@st.cache_data
def load_tfex_labels():
    """Load TranscriptFormer AI-predicted labels."""
    labels_path = Path("out_tfex/tfex_labels_for_ui.csv")
    
    if not labels_path.exists():
        return None
    
    try:
        labels_df = pd.read_csv(labels_path)
        # Ensure cell_id is string
        labels_df['cell_id'] = labels_df['cell_id'].astype(str)
        return labels_df
    except Exception as e:
        st.warning(f"Could not load TF labels: {e}")
        return None

@st.cache_data
def merge_datasets(main_df, tfex_umap_df, tfex_labels_df):
    """Merge main dataset with TranscriptFormer embeddings and labels."""
    
    # Ensure all cell_ids are strings
    main_df['cell_id'] = main_df['cell_id'].astype(str)
    
    if tfex_umap_df is None:
        return main_df, False, main_df
    
    try:
        # First, identify which cells have TF embeddings
        tfex_cells = set(tfex_umap_df['cell_id'].values)
        
        # Create subset of main data that has TF embeddings
        tfex_subset_df = main_df[main_df['cell_id'].isin(tfex_cells)].copy()
        
        # Merge TF UMAP coordinates
        tfex_subset_df = tfex_subset_df.merge(
            tfex_umap_df[['cell_id', 'umap_x', 'umap_y', 'umap_z']],
            on='cell_id',
            how='inner'
        )
        
        # Rename TF UMAP columns
        tfex_subset_df = tfex_subset_df.rename(columns={
            'umap_x': 'x_tfex',
            'umap_y': 'y_tfex',
            'umap_z': 'z_tfex'
        })
        
        # Merge TF labels if available
        if tfex_labels_df is not None:
            tfex_subset_df = tfex_subset_df.merge(
                tfex_labels_df,
                on='cell_id',
                how='left'
            )
            
            # Fill any missing labels with 'unknown' and confidence with 0
            label_cols = ['tissue_tfex_knn', 'cell_type_broad_tfex_knn']
            conf_cols = ['tissue_tfex_conf', 'cell_type_broad_tfex_conf']
            
            for col in label_cols:
                if col in tfex_subset_df.columns:
                    tfex_subset_df[col] = tfex_subset_df[col].fillna('unknown')
            
            for col in conf_cols:
                if col in tfex_subset_df.columns:
                    tfex_subset_df[col] = tfex_subset_df[col].fillna(0)
        
        tfex_available = len(tfex_subset_df) > 0
        
        if tfex_available:
            st.sidebar.success(f"✅ TF-Exemplar data loaded: {len(tfex_subset_df):,} cells")
        
        return main_df, tfex_available, tfex_subset_df
        
    except Exception as e:
        st.error(f"Failed to merge TF data: {e}")
        return main_df, False, main_df

@st.cache_data
def calculate_impact_metrics(df):
    """Calculate tissue impact metrics based on Z-score distributions."""
    
    # Calculate median Z-score for tfap2a cells by tissue
    tfap2a_stats = df[df['condition'] == 'tfap2a'].groupby('tissue', observed=True)['perturbation_z_score'].agg([
        'median', 'mean', 'std', 'count'
    ]).round(3)
    
    tfap2a_stats['impact_score'] = tfap2a_stats['median']
    
    # Calculate percentage of cells with strong perturbation
    strong_perturbation = df[df['condition'] == 'tfap2a'].groupby('tissue', observed=True).apply(
        lambda x: (x['perturbation_z_score'] > 1.5).sum() / len(x) * 100 if len(x) > 0 else 0, 
        include_groups=False
    )
    
    if isinstance(strong_perturbation, pd.DataFrame):
        strong_perturbation = strong_perturbation.squeeze()
    
    tfap2a_stats['pct_strong_effect'] = strong_perturbation.round(1)
    
    # Sort by impact score
    impact_ranking = tfap2a_stats.sort_values('impact_score', ascending=False)
    
    return impact_ranking

@st.cache_data
def filter_data(df, filter_category, filter_value, z_score_range, impact_threshold, 
                enable_outlier_filter, buffer_sigma, base_label_col='cell_type_broad', max_points=25000):
    """Filter data based on user selections."""
    filtered_df = df.copy()

    # Apply single filter
    if filter_category == "Population" and filter_value != "All Populations":
        filtered_df = filtered_df[filtered_df['condition'] == filter_value]
    elif filter_category == "Tissue" and filter_value != "All Tissues":
        # Check if we should use TF labels
        if 'tissue_tfex_knn' in filtered_df.columns and filter_value in filtered_df['tissue_tfex_knn'].unique():
            filtered_df = filtered_df[filtered_df['tissue_tfex_knn'] == filter_value]
        else:
            filtered_df = filtered_df[filtered_df['tissue'] == filter_value]
    elif filter_category == "Cell Type" and filter_value != "All Cell Types":
        # Check if we should use TF labels
        if 'cell_type_broad_tfex_knn' in filtered_df.columns and filter_value in filtered_df['cell_type_broad_tfex_knn'].unique():
            filtered_df = filtered_df[filtered_df['cell_type_broad_tfex_knn'] == filter_value]
        else:
            filtered_df = filtered_df[filtered_df['cell_type_broad'] == filter_value]
    
    # Apply Z-score range filter (only if perturbation data exists)
    if 'perturbation_z_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['perturbation_z_score'].between(*z_score_range)]
    
    # Filter by impact threshold (only if perturbation data exists)
    if impact_threshold > 0 and 'perturbation_z_score' in df.columns:
        impact_metrics = calculate_impact_metrics(df)
        high_impact_tissues = impact_metrics[impact_metrics['impact_score'] >= impact_threshold].index.tolist()
        filtered_df = filtered_df[filtered_df['tissue'].isin(high_impact_tissues)]
    
    # Apply baseline subtraction filter (only if perturbation data exists)
    if enable_outlier_filter and base_label_col in df.columns and 'perturbation_z_score' in df.columns:
        # Use the specified label column for natural variation calculation
        control_ranges = df[df['condition'] == 'control'].groupby(base_label_col)['perturbation_z_score'].agg(['min', 'max'])
        
        tfap2a_mask = filtered_df['condition'] == 'tfap2a'
        control_mask = filtered_df['condition'] == 'control'
        
        novel_state_mask = pd.Series(False, index=filtered_df.index)
        outlier_count = 0
        
        for cell_type in control_ranges.index:
            if cell_type in filtered_df[base_label_col].values:
                cell_type_mask = (filtered_df[base_label_col] == cell_type)
                tfap2a_cell_type_mask = tfap2a_mask & cell_type_mask
                
                min_natural = control_ranges.loc[cell_type, 'min'] - buffer_sigma
                max_natural = control_ranges.loc[cell_type, 'max'] + buffer_sigma
                
                beyond_natural = (
                    tfap2a_cell_type_mask & 
                    ((filtered_df['perturbation_z_score'] > max_natural) | 
                     (filtered_df['perturbation_z_score'] < min_natural))
                )
                
                novel_state_mask |= beyond_natural
                outlier_count += beyond_natural.sum()
        
        filtered_df = filtered_df[control_mask | novel_state_mask]
        
        if outlier_count > 0:
            st.sidebar.success(f"🔬 Found {outlier_count:,} tfap2a cells with novel states")
    
    # Sample for performance
    if len(filtered_df) > max_points:
        filtered_df = filtered_df.sample(n=max_points, random_state=42)
        st.sidebar.warning(f"⚠️ Showing random sample of {max_points:,} cells")
        
    return filtered_df

def create_enhanced_3d_plot(df, full_df, embedding_method='classical', color_by='perturbation_z_score', 
                           opacity=0.7, point_size=2.5, highlight_extreme=True, 
                           show_ghost=False, ghost_opacity=0.1, title_suffix=""):
    """Create enhanced 3D plot highlighting perturbation effects."""
    if df.empty:
        return None

    # Increase sampling for better visualization
    max_points = 20000 if embedding_method == 'tfex' else 25000
    plot_df = df.sample(n=min(len(df), max_points), random_state=42)
    sample_note = f"Showing {len(plot_df):,} cells"

    # Select coordinates based on embedding method
    if embedding_method == 'tfex' and all(col in plot_df.columns for col in ['x_tfex', 'y_tfex', 'z_tfex']):
        x_col, y_col, z_col = 'x_tfex', 'y_tfex', 'z_tfex'
        method_label = "TF-Exemplar"
        title_prefix = "AI Landscape"
    else:
        x_col, y_col, z_col = 'x', 'y', 'z'
        method_label = "Classical"
        title_prefix = "Classical Landscape"

    fig = go.Figure()
    
    # Add ghost layer if enabled
    if show_ghost:
        ghost_df = full_df.sample(n=min(len(full_df), 15000), random_state=42)
        if embedding_method == 'tfex' and all(col in ghost_df.columns for col in ['x_tfex', 'y_tfex', 'z_tfex']):
            ghost_x, ghost_y, ghost_z = ghost_df['x_tfex'], ghost_df['y_tfex'], ghost_df['z_tfex']
        else:
            ghost_x, ghost_y, ghost_z = ghost_df['x'], ghost_df['y'], ghost_df['z']
            
        fig.add_trace(go.Scatter3d(
            x=ghost_x, y=ghost_y, z=ghost_z,
            mode='markers',
            marker=dict(size=point_size * 0.5, color='lightgray', opacity=ghost_opacity, line=dict(width=0)),
            name='All Cells (Ghost)',
            hoverinfo='skip',
            showlegend=False
        ))

    # Main plotting logic
    if color_by == 'perturbation_z_score' and 'perturbation_z_score' in plot_df.columns:
        plot_df['z_score_display'] = plot_df['perturbation_z_score'].copy()
        
        if highlight_extreme:
            plot_df['point_size'] = np.where(
                abs(plot_df['perturbation_z_score']) > 2.0, point_size * 1.5, point_size
            )
            size_array = plot_df['point_size']
        else:
            size_array = point_size
            
        # Build hover text
        hover_texts = []
        for _, row in plot_df.iterrows():
            hover_text = f"Z-Score: {row['perturbation_z_score']:.2f}"
            
            if 'condition' in row:
                hover_text = f"Condition: {row['condition']}<br>" + hover_text
            
            # Add appropriate labels based on what's available
            if 'tissue_tfex_knn' in row and pd.notna(row['tissue_tfex_knn']):
                hover_text += f"<br>Tissue (AI): {row['tissue_tfex_knn']}"
                if 'tissue_tfex_conf' in row:
                    hover_text += f" (conf: {row['tissue_tfex_conf']:.2f})"
            else:
                hover_text += f"<br>Tissue: {row['tissue']}"
            
            if 'cell_type_broad_tfex_knn' in row and pd.notna(row['cell_type_broad_tfex_knn']):
                hover_text += f"<br>Cell Type (AI): {row['cell_type_broad_tfex_knn']}"
                if 'cell_type_broad_tfex_conf' in row:
                    hover_text += f" (conf: {row['cell_type_broad_tfex_conf']:.2f})"
            else:
                hover_text += f"<br>Cell Type: {row['cell_type_broad']}"
            
            hover_texts.append(hover_text)
            
        fig.add_trace(go.Scatter3d(
            x=plot_df[x_col], y=plot_df[y_col], z=plot_df[z_col],
            mode='markers',
            marker=dict(
                size=size_array,
                color=plot_df['z_score_display'],
                colorscale='RdBu_r',
                cmin=-3, cmax=3,
                opacity=opacity,
                line=dict(width=0),
                colorbar=dict(
                    title="Perturbation<br>Z-Score",
                    tickvals=[-3, -2, -1, 0, 1, 2, 3],
                    ticktext=["-3σ", "-2σ", "-1σ", "0", "+1σ", "+2σ", "+3σ"],
                    x=1.02, len=0.8
                )
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name='Cells',
            showlegend=False
        ))
        
    else:
        # Categorical coloring - show all cells by category
        plot_df['hover_z_score'] = plot_df.get('perturbation_z_score', 'N/A')
        
        for category in plot_df[color_by].unique():
            if pd.notna(category):
                cat_df = plot_df[plot_df[color_by] == category]
                
                # Build hover text for categorical
                hover_texts = []
                for _, row in cat_df.iterrows():
                    hover_text = f"{color_by.replace('_tfex_knn', ' (AI)').replace('_', ' ').title()}: {category}"
                    
                    if 'condition' in row:
                        hover_text += f"<br>Condition: {row['condition']}"
                    
                    if isinstance(row['hover_z_score'], (int, float)) and not pd.isna(row['hover_z_score']):
                        hover_text += f"<br>Z-Score: {row['hover_z_score']:.2f}"
                    
                    hover_texts.append(hover_text)
                
                fig.add_trace(go.Scatter3d(
                    x=cat_df[x_col], y=cat_df[y_col], z=cat_df[z_col],
                    mode='markers',
                    marker=dict(size=point_size, opacity=opacity, line=dict(width=0)),
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>',
                    name=str(category),
                    showlegend=True
                ))
    
    # Update layout
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
        height=1300,
        margin=dict(l=0, r=0, t=60, b=0),
        title=f"{title_prefix} {title_suffix} {sample_note}",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(14, 17, 23, 0.8)', bordercolor='rgba(255, 255, 255, 0.2)', borderwidth=1)
    )
    
    return fig

def create_impact_violin_plot(df, top_n=12):
    """Create split violin plots showing perturbation distribution by tissue."""
    
    impact_metrics = calculate_impact_metrics(df)
    top_tissues = impact_metrics.head(top_n).index.tolist()
    
    plot_df = df[df['tissue'].isin(top_tissues)].copy()
    
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
                    showlegend=(i == 0),
                    legendgroup=condition,
                    scalegroup=condition
                ))
    
    fig.update_layout(
        title="Perturbation Impact by Tissue",
        xaxis_title="Tissue (Ranked by Impact)",
        yaxis_title="Perturbation Z-Score",
        violinmode='overlay',
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        height=1300,
        xaxis=dict(tickangle=45, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.3)')
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5, annotation_text="Baseline")
    fig.add_hline(y=2, line_dash="dot", line_color="orange", opacity=0.5, annotation_text="+2σ")
    fig.add_hline(y=-2, line_dash="dot", line_color="orange", opacity=0.5)
    
    return fig

def display_impact_summary(df):
    """Display summary metrics of perturbation impact."""
    
    impact_metrics = calculate_impact_metrics(df)
    
    if len(impact_metrics) == 0:
        st.warning("No impact metrics available")
        return
    
    st.markdown("### 🎯 Tissue Impact Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        most_impacted = impact_metrics.index[0] if len(impact_metrics) > 0 else 'N/A'
        impact_score = impact_metrics.iloc[0]['impact_score'] if len(impact_metrics) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Most Impacted</h3>
            <h2>{most_impacted}</h2>
            <p>Z-score: {impact_score:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        strong_effect_tissues = (impact_metrics['pct_strong_effect'] > 20).sum() if len(impact_metrics) > 0 else 0
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
        median_shift = 0 if pd.isna(median_shift) else median_shift
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Median Shift</h3>
            <h2>{median_shift:.2f}σ</h2>
            <p>from baseline</p>
        </div>
        """, unsafe_allow_html=True)

def create_dual_interface(df, tfex_subset_df, tfex_available):
    """Create side-by-side comparison with separate controls for each view."""
    
    st.markdown(f"### 🌌 Side-by-Side Landscape Comparison", unsafe_allow_html=True)
    
    # Create two main columns for the plots
    left_col, right_col = st.columns(2)
    
    # Create two sidebar columns for controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("## 🔀 Dual View Controls")
        
        left_sidebar, right_sidebar = st.columns(2)
        
        # LEFT SIDE CONTROLS (Classical)
        with left_sidebar:
            st.markdown("### 🔬 Classical Controls")
            
            # Classical color options
            classical_color_options = ['perturbation_z_score', 'condition', 'tissue', 'cell_type_broad']
            classical_color_by = st.selectbox(
                "Classical Color By", 
                classical_color_options,
                index=0,
                format_func=lambda x: x.replace('_', ' ').title(),
                key="classical_color"
            )
            
            # Classical filters
            classical_filter_category = st.selectbox(
                "Classical Filter",
                ["Population", "Tissue", "Cell Type"],
                key="classical_filter_cat"
            )
            
            if classical_filter_category == "Population":
                unique_values = sorted([x for x in df['condition'].unique() if pd.notna(x)])
            elif classical_filter_category == "Tissue":
                unique_values = sorted([x for x in df['tissue'].unique() if pd.notna(x)])
            else:
                unique_values = sorted([x for x in df['cell_type_broad'].unique() if pd.notna(x)])
            
            classical_filter_options = [f"All {classical_filter_category}s"] + unique_values
            classical_filter_value = st.selectbox(
                f"Select {classical_filter_category}",
                classical_filter_options,
                index=0,
                key="classical_filter_val"
            )
            
            # Classical Z-score range
            min_z, max_z = float(df['perturbation_z_score'].min()), float(df['perturbation_z_score'].max())
            classical_z_range = st.slider(
                "Classical Z-Score", 
                min_z, max_z, (min_z, max_z),
                key="classical_z_range"
            )
            
            # Classical impact threshold
            classical_impact = st.slider(
                "Classical Impact", 
                0.0, 2.0, 0.0, 0.1,
                key="classical_impact"
            )
            
        # RIGHT SIDE CONTROLS (AI)
        with right_sidebar:
            st.markdown("### 🤖 AI Controls")
            
            # AI color options (limited set)
            ai_color_options = ['perturbation_z_score']
            if 'tissue_tfex_knn' in tfex_subset_df.columns:
                ai_color_options.append('tissue_tfex_knn')
            if 'cell_type_broad_tfex_knn' in tfex_subset_df.columns:
                ai_color_options.append('cell_type_broad_tfex_knn')
            
            ai_color_by = st.selectbox(
                "AI Color By", 
                ai_color_options,
                index=0,
                format_func=lambda x: x.replace('_tfex_knn', ' (AI)').replace('_', ' ').title(),
                key="ai_color"
            )
            
            # AI confidence controls
            if '_tfex_knn' in ai_color_by:
                ai_min_conf = st.slider(
                    "AI Confidence", 
                    0.0, 1.0, 0.50, 0.05,
                    key="ai_conf"
                )
                ai_hide_unknown = st.checkbox(
                    "Hide 'unknown'", 
                    value=True,
                    key="ai_hide_unknown"
                )
            else:
                ai_min_conf, ai_hide_unknown = 0.0, False
            
            # AI filters (no population filter since TF is control-only)
            ai_filter_category = st.selectbox(
                "AI Filter",
                ["Tissue", "Cell Type"],
                key="ai_filter_cat"
            )
            
            if ai_filter_category == "Tissue" and 'tissue_tfex_knn' in tfex_subset_df.columns:
                unique_values = sorted([x for x in tfex_subset_df['tissue_tfex_knn'].unique() if pd.notna(x) and x != 'unknown'])
            elif ai_filter_category == "Cell Type" and 'cell_type_broad_tfex_knn' in tfex_subset_df.columns:
                unique_values = sorted([x for x in tfex_subset_df['cell_type_broad_tfex_knn'].unique() if pd.notna(x) and x != 'unknown'])
            elif ai_filter_category == "Tissue":
                unique_values = sorted([x for x in tfex_subset_df['tissue'].unique() if pd.notna(x)])
            else:
                unique_values = sorted([x for x in tfex_subset_df['cell_type_broad'].unique() if pd.notna(x)])
            
            ai_filter_options = [f"All {ai_filter_category}s"] + unique_values
            ai_filter_value = st.selectbox(
                f"Select {ai_filter_category}",
                ai_filter_options,
                index=0,
                key="ai_filter_val"
            )
    
    # Shared visual controls
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎨 Shared Visual Settings")
        
        dual_point_size = st.slider("Point Size", 0.5, 8.0, 2.5, 0.5, key="dual_point_size")
        dual_opacity = st.slider("Opacity", 0.1, 1.0, 0.7, 0.05, key="dual_opacity")
        dual_highlight = st.checkbox("Highlight Extreme Effects", value=True, key="dual_highlight")
        dual_ghost = st.checkbox("Show Ghost Layer", value=False, key="dual_ghost")
        
        if dual_ghost:
            dual_ghost_opacity = st.slider("Ghost Opacity", 0.01, 0.3, 0.1, 0.01, key="dual_ghost_opacity")
        else:
            dual_ghost_opacity = 0.1
    
    # Process Classical Data
    classical_working_df = df.copy()
    classical_filtered_df = filter_data(
        classical_working_df, 
        classical_filter_category, 
        classical_filter_value, 
        classical_z_range, 
        classical_impact,
        False,  # No baseline subtraction for simplicity
        0.1, 
        base_label_col='cell_type_broad'
    )
    
    # Process AI Data
    ai_working_df = tfex_subset_df.copy()
    
    # Apply AI confidence filtering
    if '_tfex_knn' in ai_color_by:
        conf_col = ai_color_by.replace('_tfex_knn', '_tfex_conf')
        if conf_col in ai_working_df.columns:
            ai_working_df = ai_working_df[ai_working_df[conf_col] >= ai_min_conf]
            if ai_hide_unknown:
                ai_working_df = ai_working_df[ai_working_df[ai_color_by] != 'unknown']
    
    ai_filtered_df = filter_data(
        ai_working_df, 
        ai_filter_category, 
        ai_filter_value, 
        (-10, 10),  # No meaningful perturbation filtering for TF
        0.0,
        False,  # No baseline subtraction
        0.1, 
        base_label_col='cell_type_broad'
    )
    
    # Display the plots
    with left_col:
        st.markdown(f"#### <span class='embedding-badge-classical'>🔬 Classical</span>", unsafe_allow_html=True)
        st.caption(f"📊 {len(classical_filtered_df):,} cells")
        
        fig_classical = create_enhanced_3d_plot(
            classical_filtered_df, 
            full_df=classical_working_df,
            embedding_method='classical',
            color_by=classical_color_by,
            opacity=dual_opacity,
            point_size=dual_point_size,
            highlight_extreme=dual_highlight,
            show_ghost=dual_ghost,
            ghost_opacity=dual_ghost_opacity,
            title_suffix=""
        )
        
        if fig_classical:
            fig_classical.update_layout(height=1300)
            st.plotly_chart(fig_classical, use_container_width=True)
    
    with right_col:
        st.markdown(f"#### <span class='embedding-badge-ai'>🤖 AI-Powered</span>", unsafe_allow_html=True)
        st.caption(f"📊 {len(ai_filtered_df):,} cells")
        
        fig_ai = create_enhanced_3d_plot(
            ai_filtered_df, 
            full_df=ai_working_df,
            embedding_method='tfex',
            color_by=ai_color_by,
            opacity=dual_opacity,
            point_size=dual_point_size,
            highlight_extreme=dual_highlight,
            show_ghost=dual_ghost,
            ghost_opacity=dual_ghost_opacity,
            title_suffix=""
        )
        
        if fig_ai:
            fig_ai.update_layout(height=1300)
            st.plotly_chart(fig_ai, use_container_width=True)
        else:
            st.warning(f"No AI data available for current filters.")
    
    # Interpretation guide
    with st.container():
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **🔬 Classical Interpretation:**
            - Colored by: {classical_color_by.replace('_', ' ').title()}
            - Uses traditional PCA/UMAP dimensionality reduction
            - {len(classical_filtered_df):,} cells displayed
            """)
        
        with col2:
            st.markdown(f"""
            **🤖 AI Interpretation:**
            - Colored by: {ai_color_by.replace('_tfex_knn', ' (AI)').replace('_', ' ').title()}
            - Uses TranscriptFormer foundation model embeddings
            - {len(ai_filtered_df):,} cells displayed
            """)

# --- Main App ---
def main():
    # Clear cache
    st.cache_data.clear()
    
    # Title
    st.markdown("""
    <h1 class='main-title'>🎯 ZSCAPE Perturbation Impact Explorer</h1>
    <p class='subtitle'>
        Visualizing tfap2a knockout effects • Classical analysis + TranscriptFormer AI embeddings
    </p>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading scored perturbation data..."):
        df = load_scored_data()
    
    # Load TranscriptFormer outputs
    with st.spinner("Loading TranscriptFormer outputs..."):
        tfex_umap_df = load_tfex_data()
        tfex_labels_df = load_tfex_labels()
        df, tfex_available, tfex_subset_df = merge_datasets(df, tfex_umap_df, tfex_labels_df)
    
    # Get filter options
    populations = sorted([x for x in df['condition'].unique() if pd.notna(x)])
    tissues = sorted([x for x in df['tissue'].unique() if pd.notna(x)])
    cell_types = sorted([x for x in df['cell_type_broad'].unique() if pd.notna(x)])
    
    # --- Sidebar Controls ---
    st.sidebar.markdown("## 🎛️ Visualization Controls")

    # Embedding method selection
    if tfex_available:
        embedding_options = ['classical', 'tfex', 'both']
        embedding_labels = {
            'classical': "🔬 Classical PCA/UMAP", 
            'tfex': "🤖 TF-Exemplar (AI)",
            'both': "🔀 Side-by-Side Comparison"
        }
        embedding_method = st.sidebar.selectbox(
            "🧠 Embedding Method",
            embedding_options,
            format_func=lambda x: embedding_labels[x],
            help="Choose between classical dimensionality reduction, TranscriptFormer AI embeddings, or side-by-side comparison"
        )
        
        if embedding_method == 'tfex':
            st.sidebar.markdown("""
            <div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 8px; border-radius: 8px; margin: 5px 0;'>
                <b>🤖 AI-Powered Analysis</b><br>
                <small>Using TranscriptFormer foundation model</small>
            </div>
            """, unsafe_allow_html=True)
        elif embedding_method == 'both':
            st.sidebar.markdown("""
            <div style='background: linear-gradient(45deg, #4a5568, #ff6b6b, #4ecdc4); padding: 8px; border-radius: 8px; margin: 5px 0;'>
                <b>🔀 Dual Analysis</b><br>
                <small>Classical vs AI side-by-side</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        embedding_method = 'classical'
        st.sidebar.info("🔬 Using Classical embeddings")

    # Color selection based on embedding method
    if embedding_method == 'tfex':
        # TF mode: only AI-specific options
        color_options = ['perturbation_z_score']
        if 'tissue_tfex_knn' in tfex_subset_df.columns:
            color_options.append('tissue_tfex_knn')
        if 'cell_type_broad_tfex_knn' in tfex_subset_df.columns:
            color_options.append('cell_type_broad_tfex_knn')
    else:
        # Classical or both modes: all options
        color_options = ['perturbation_z_score', 'condition', 'tissue', 'cell_type_broad']
        
        # Add TF label options if available
        if embedding_method == 'both' and tfex_available:
            if 'tissue_tfex_knn' in tfex_subset_df.columns:
                color_options.append('tissue_tfex_knn')
            if 'cell_type_broad_tfex_knn' in tfex_subset_df.columns:
                color_options.append('cell_type_broad_tfex_knn')
    
    color_by = st.sidebar.selectbox(
        "Color Points By", 
        color_options,
        index=0,
        format_func=lambda x: x.replace('_tfex_knn', ' (AI predicted)').replace('_', ' ').title()
    )
    
    # TF confidence controls when using AI labels
    if tfex_available and '_tfex_knn' in color_by:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧠 AI Label Controls")
        min_conf = st.sidebar.slider(
            "Min confidence threshold", 
            0.0, 1.0, 0.50, 0.05,
            help="Only show cells where AI confidence ≥ threshold"
        )
        hide_unknown = st.sidebar.checkbox(
            "Hide 'unknown' labels", 
            value=True,
            help="Hide cells where AI couldn't predict a label"
        )
    else:
        min_conf, hide_unknown = 0.0, False
    
    # Visualization options
    highlight_extreme = st.sidebar.checkbox(
        "Highlight Extreme Effects", 
        value=True,
        help="Make cells with |Z-score| > 2 larger"
    )
    
    point_size = st.sidebar.slider(
        "Point Size", 
        0.5, 8.0, 2.5, 0.5
    )
    
    show_ghost = st.sidebar.checkbox(
        "Show Ghost Layer", 
        value=False,
        help="Show all cells as background"
    )
    
    if show_ghost:
        ghost_opacity = st.sidebar.slider(
            "Ghost Layer Opacity", 
            0.01, 0.3, 0.1, 0.01
        )
    else:
        ghost_opacity = 0.1
    
    # Filtering controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters")
    
    # Population filter - disabled in TF mode
    if embedding_method != 'tfex':
        filter_category = st.sidebar.selectbox(
            "Filter Category",
            ["Population", "Tissue", "Cell Type"]
        )
    else:
        # In TF mode, skip population filtering
        filter_category = st.sidebar.selectbox(
            "Filter Category",
            ["Tissue", "Cell Type"]
        )
    
    # Get filter options based on embedding method
    if embedding_method == 'tfex' and tfex_available:
        # Use TF labels if available
        if filter_category == "Tissue" and 'tissue_tfex_knn' in tfex_subset_df.columns:
            unique_values = sorted([x for x in tfex_subset_df['tissue_tfex_knn'].unique() if pd.notna(x) and x != 'unknown'])
        elif filter_category == "Cell Type" and 'cell_type_broad_tfex_knn' in tfex_subset_df.columns:
            unique_values = sorted([x for x in tfex_subset_df['cell_type_broad_tfex_knn'].unique() if pd.notna(x) and x != 'unknown'])
        elif filter_category == "Population":
            unique_values = populations
        elif filter_category == "Tissue":
            unique_values = tissues
        else:
            unique_values = cell_types
    else:
        if filter_category == "Population":
            unique_values = populations
        elif filter_category == "Tissue":
            unique_values = tissues
        else:
            unique_values = cell_types
    
    filter_options = [f"All {filter_category}s"] + unique_values
    filter_value = st.sidebar.selectbox(
        f"Select {filter_category}",
        filter_options,
        index=0
    )
    
    # Perturbation-specific filters - only show in classical/both modes
    if embedding_method != 'tfex':
        # Z-score range filter
        min_z, max_z = float(df['perturbation_z_score'].min()), float(df['perturbation_z_score'].max())
        z_score_range = st.sidebar.slider(
            "Z-Score Range", 
            min_z, max_z, (min_z, max_z),
            help="Filter by perturbation intensity"
        )
        
        # Impact threshold
        impact_threshold = st.sidebar.slider(
            "Min Tissue Impact", 
            0.0, 2.0, 0.0, 0.1,
            help="Show tissues with median Z ≥ threshold"
        )
        
        # Baseline subtraction
        enable_outlier_filter = st.sidebar.checkbox(
            "Baseline Subtraction", 
            value=False,
            help="Show only tfap2a cells beyond natural variation"
        )
        
        if enable_outlier_filter:
            buffer_sigma = st.sidebar.slider(
                "Buffer (σ)", 
                0.0, 1.0, 0.1, 0.05
            )
            
            # Choose which labels to use for baseline calculation
            if embedding_method == 'both' and 'cell_type_broad_tfex_knn' in tfex_subset_df.columns:
                use_tf_baseline = st.sidebar.checkbox(
                    "Use AI labels for baseline", 
                    value=True,
                    help="Calculate natural variation using AI-predicted cell types"
                )
                base_label_col = 'cell_type_broad_tfex_knn' if use_tf_baseline else 'cell_type_broad'
            else:
                base_label_col = 'cell_type_broad'
        else:
            buffer_sigma = 0.1
            base_label_col = 'cell_type_broad'
    else:
        # TF mode defaults
        z_score_range = (-10, 10)  # No meaningful perturbation data
        impact_threshold = 0.0
        enable_outlier_filter = False
        buffer_sigma = 0.1
        base_label_col = 'cell_type_broad'

    st.sidebar.markdown("---")
    opacity = st.sidebar.slider("Point Opacity", 0.1, 1.0, 0.7, 0.05)

    # Data info with debugging
    st.sidebar.markdown("---")
    if embedding_method == 'tfex' and tfex_available:
        st.sidebar.info(f"""
        **Mode:** 🤖 TF-Exemplar AI
        **TF Cells:** {len(tfex_subset_df):,}
        **Control Only:** {(tfex_subset_df['condition'] == 'control').sum():,}
        """)
    elif embedding_method == 'both':
        tfex_count = len(working_df_tfex) if 'working_df_tfex' in locals() else len(tfex_subset_df)
        st.sidebar.info(f"""
        **Mode:** 🔀 Dual View
        **Classical:** {len(df):,} cells
        **TF-Exemplar:** {tfex_count:,} cells
        """)
        
        # Show tissue label status for debugging
        if 'working_df_tfex' in locals():
            unique_tissues = working_df_tfex['tissue'].nunique() if 'tissue' in working_df_tfex.columns else 0
            unique_ai_tissues = working_df_tfex['tissue_tfex_knn'].nunique() if 'tissue_tfex_knn' in working_df_tfex.columns else 0
            st.sidebar.caption(f"AI dataset: {unique_tissues} tissues, {unique_ai_tissues} AI tissues")
        
        # Debug info for TF cell count issue
        if tfex_count < 10000:
            st.sidebar.warning(f"""
            ⚠️ **Limited TF Data**
            
            TF-Exemplar only processed {tfex_count:,} cells.
            Check if `out_tfex/tfex_umap3d_for_ui.csv` contains the expected number of rows.
            
            Expected: ~20k+ cells
            Actual: {tfex_count:,} cells
            """)
    else:
        st.sidebar.info(f"""
        **Mode:** 🔬 Classical
        **Total Cells:** {len(df):,}
        **Control:** {(df['condition'] == 'control').sum():,}
        **tfap2a:** {(df['condition'] == 'tfap2a').sum():,}
        """)

    # --- Data Processing ---
    if embedding_method == 'both':
        # Data processing is handled within create_dual_interface
        pass
    else:
        # Use appropriate dataset for single view modes
        if embedding_method == 'tfex' and tfex_available:
            working_df = tfex_subset_df.copy()
            
            # Apply confidence filtering if using TF labels
            if '_tfex_knn' in color_by:
                conf_col = color_by.replace('_tfex_knn', '_tfex_conf')
                if conf_col in working_df.columns:
                    pre_filter = len(working_df)
                    working_df = working_df[working_df[conf_col] >= min_conf]
                    
                    if hide_unknown:
                        working_df = working_df[working_df[color_by] != 'unknown']
                    
                    post_filter = len(working_df)
                    if post_filter < pre_filter:
                        st.info(f"Filtered to {post_filter:,} cells (confidence ≥ {min_conf:.2f})")
        else:
            working_df = df.copy()
        
        # Apply filters
        filtered_df = filter_data(
            working_df, 
            filter_category, 
            filter_value, 
            z_score_range, 
            impact_threshold,
            enable_outlier_filter, 
            buffer_sigma, 
            base_label_col=base_label_col
        )

        if filtered_df.empty:
            st.warning("No cells match the selected filters.")
            return

    # --- Main Visualization ---
    if embedding_method == 'both':
        # Use the new dual interface
        create_dual_interface(df, tfex_subset_df, tfex_available)
        
    else:
        # Single view
        if embedding_method == 'tfex':
            method_badge = "🤖 AI-Powered"
            badge_class = "embedding-badge-ai"
        else:
            method_badge = "🔬 Classical"
            badge_class = "embedding-badge-classical"
            
        st.markdown(f"### 🌌 3D Landscape <span class='{badge_class}'>{method_badge}</span>", unsafe_allow_html=True)
        
        fig_3d = create_enhanced_3d_plot(
            filtered_df, 
            full_df=working_df,
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
    
    # Only show interpretation and impact analysis for single view modes
    if embedding_method != 'both':
        # Interpretation guide
        if color_by == 'perturbation_z_score':
            interpretation_text = """
            **🎯 Interpretation:**
            - **Red**: Strong positive perturbation (increased expression/activity)
            - **Blue**: Strong negative perturbation (decreased expression/activity)  
            - **White**: Near baseline (minimal effect)
            """
            
            if embedding_method in ['tfex']:
                interpretation_text += """
            - **AI Enhancement**: TF embeddings may reveal hidden perturbation patterns
            """
                
        elif '_tfex_knn' in color_by:
            label_type = color_by.replace('_tfex_knn', '').replace('_', ' ').title()
            interpretation_text = f"""
            **🤖 AI-Predicted {label_type}:**
            - Labels predicted by k-NN on TranscriptFormer embeddings
            - Confidence threshold: {min_conf:.2f}
            - Unknown cells {'hidden' if hide_unknown else 'shown in gray'}
            """
        else:
            interpretation_text = f"""
            **Colored by:** {color_by.replace('_', ' ').title()}
            """
        
        st.markdown(interpretation_text)
        
        # --- Impact Analysis (only for perturbation-based modes) ---
        if embedding_method != 'tfex':
            st.markdown("---")
            
            # Use full dataset for impact metrics (not filtered)
            display_impact_summary(df)
            
            st.markdown("### 🎻 Tissue Impact Distribution")
            violin_fig = create_impact_violin_plot(df)
            st.plotly_chart(violin_fig, use_container_width=True)
    
    # --- TF Integration Details ---
    if tfex_available:
        with st.expander("🔧 TranscriptFormer Integration Details"):
            
            # Load and display label report if available
            report_path = Path("out_tfex/tfex_label_report.json")
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                
                st.markdown("### 📊 AI Label Transfer Performance")
                
                st.markdown("""
                **What the AI is actually doing:**
                
                The AI side isn't magic - it's using a pre-trained foundation model (TranscriptFormer) that learned patterns from millions of cells. Here's the simple breakdown:
                
                1. **Smart Pattern Recognition**: The foundation model takes your cell's gene expression and converts it into a "biological fingerprint" - a high-dimensional vector that captures what the cell is doing biologically, not just which genes are turned on/off.
                
                2. **Neighborhood Voting**: For labels, we train a simple k-nearest neighbors classifier using cells we already know the identity of (control cells with known tissue/cell type). When predicting a new cell, it finds the most similar cells in the AI embedding space and asks "what are my neighbors?"
                
                3. **Confidence Scoring**: The confidence comes from how unanimous the neighborhood vote is. If 8 out of 10 nearest neighbors agree it's "muscle tissue," confidence is high. If it's split 5/5 between two types, confidence is low.
                
                4. **Foundation Model Advantage**: Because TranscriptFormer learned from diverse datasets, it can recognize biological patterns that might be invisible to classical PCA-based approaches, especially for rare cell types or transitional states.
                
                The result: AI-predicted labels that often align remarkably well with expert annotations, plus the ability to confidently label cells that classical methods might struggle with.
                """)
                
                col1, col2 = st.columns(2)
                
                for i, (label, stats) in enumerate(report.get("labels", {}).items()):
                    target_col = col1 if i == 0 else col2
                    with target_col:
                        st.markdown(f"""
                        **{label.replace('_', ' ').title()}:**
                        - Training cells: {stats.get('train_cells', 'N/A'):,}
                        - Classes: {stats.get('n_classes_used', 'N/A')}
                        - CV accuracy: {stats.get('cv_acc_mean', 0):.1%}
                        - Unknown rate: {stats.get('unknown_rate', 0):.1%}
                        """)
            
            st.markdown("""
            ### 🧬 TranscriptFormer Foundation Model Analysis
            
            **What powers the AI embeddings:**
            
            **1. Model Input**: TranscriptFormer received the exact same single-cell RNA-seq expression profiles that went into the classical ZSCAPE analysis. Each cell's complete transcriptomic signature was fed into the foundation model without any preprocessing or dimensionality reduction.
            
            **2. Foundation Model Inference**: The pre-trained TranscriptFormer model (trained on millions of cells across diverse tissues and conditions) generated rich 2048-dimensional embeddings that capture fundamental biological relationships and cellular states learned from its massive training corpus.
            
            **3. UMAP Projection**: The high-dimensional TF embeddings were projected into 3D space using UMAP, preserving local neighborhood structure while making visualization possible.
            
            **4. AI Label Transfer**: k-NN classifiers trained on TF embeddings predict tissue and cell type labels, leveraging the model's learned biological knowledge to make confident predictions even for rare or transitional cell states.
            
            **5. Perturbation Visualization**: The same ZSCAPE perturbation Z-scores are mapped onto the AI embedding space, revealing how perturbation effects appear when viewed through the lens of a foundation model trained on diverse biological contexts.
            
            **Why it works so well:**
            - **Biological Priors**: TranscriptFormer learned fundamental gene regulatory patterns from its training data
            - **Context Awareness**: The model captures complex gene-gene interactions and pathway relationships
            - **Noise Robustness**: Foundation model embeddings are less sensitive to technical noise and batch effects
            - **Cross-Dataset Generalization**: Training on diverse datasets helps the model recognize conserved biological patterns
            
            The remarkable concordance between AI and classical cell type assignments demonstrates that TranscriptFormer has internalized genuine biological relationships, not just technical artifacts.
            """)

if __name__ == "__main__":
    main()