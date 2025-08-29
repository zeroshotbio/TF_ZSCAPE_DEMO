import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="ZSCAPE Control Populations Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stSelectbox > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    .stSlider > div > div > div {
        color: #fafafa;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stMultiSelect > div > div {
        background-color: #262730;
        color: #fafafa;
    }
    /* Expand main content area */
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
    }
    /* Make tabs more prominent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        border-radius: 5px;
        color: #fafafa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_control_data():
    """Load prepared control population data."""
    data_path = Path("demo_data/zscape_controls_viz.csv")
    
    if not data_path.exists():
        st.error(f"""
        Control data not found: {data_path}
        
        Please run the data preparation first:
        ```python
        python prepare_controls.py
        ```
        """)
        st.stop()
    
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def get_filter_options(df):
    """Get available options for filtering."""
    tissues = sorted(df['tissue'].unique())
    timepoints = sorted(df['timepoint'].unique())
    cell_types_broad = sorted(df['cell_type_broad'].unique())
    
    return tissues, timepoints, cell_types_broad

@st.cache_data
def filter_control_data(df, selected_tissues, selected_timepoints, selected_cell_types,
                       umi_range, gene_range, max_points=50000):
    """Filter control data based on selections."""
    
    # Apply filters
    filtered_df = df.copy()
    
    if "All Tissues" not in selected_tissues:
        filtered_df = filtered_df[filtered_df['tissue'].isin(selected_tissues)]
    
    if "All Timepoints" not in selected_timepoints:
        timepoint_values = []
        for tp in selected_timepoints:
            if 'hpf' in str(tp):
                timepoint_values.append(int(str(tp).split()[0]))
            else:
                timepoint_values.append(tp)
        filtered_df = filtered_df[filtered_df['timepoint'].isin(timepoint_values)]
    
    if "All Cell Types" not in selected_cell_types:
        filtered_df = filtered_df[filtered_df['cell_type_broad'].isin(selected_cell_types)]
    
    # Apply UMI and gene count filters
    filtered_df = filtered_df[
        (filtered_df['n_umi'] >= umi_range[0]) & 
        (filtered_df['n_umi'] <= umi_range[1]) &
        (filtered_df['n_genes'] >= gene_range[0]) & 
        (filtered_df['n_genes'] <= gene_range[1])
    ]
    
    # Sample for performance if needed
    if len(filtered_df) > max_points:
        filtered_df = filtered_df.sample(n=max_points, random_state=42)
        st.sidebar.warning(f"⚠️ Showing random sample of {max_points:,} cells for performance")
    
    return filtered_df

def create_3d_scatter_plot(df, color_by='tissue', size_by='n_umi', opacity=0.7):
    """Create 3D scatter plot of control populations."""
    
    if len(df) == 0:
        return None
    
    # Sample further if still too many points for 3D
    if len(df) > 30000:
        plot_df = df.sample(n=30000, random_state=42)
        sample_note = f"(showing {30000:,} of {len(df):,} cells)"
    else:
        plot_df = df
        sample_note = f"({len(df):,} cells)"
    
    # Determine point sizes
    if size_by == 'n_umi':
        sizes = np.log10(plot_df['n_umi'] + 1) * 2
        size_label = "log10(UMI count)"
    elif size_by == 'n_genes':
        sizes = np.log10(plot_df['n_genes'] + 1) * 2
        size_label = "log10(Gene count)"
    else:
        sizes = [3] * len(plot_df)
        size_label = "Fixed size"
    
    # Create 3D scatter plot
    if color_by == 'tissue':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='tissue',
            hover_data=['cell_type_broad', 'timepoint', 'n_umi', 'n_genes'],
            title=f"ZSCAPE Control Populations in 3D Space {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    elif color_by == 'cell_type_broad':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='cell_type_broad',
            hover_data=['tissue', 'timepoint', 'n_umi', 'n_genes'],
            title=f"ZSCAPE Control Populations by Cell Type {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    elif color_by == 'timepoint':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='timepoint',
            hover_data=['tissue', 'cell_type_broad', 'n_umi', 'n_genes'],
            title=f"ZSCAPE Control Populations by Timepoint {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_continuous_scale='viridis'
        )
    elif color_by == 'n_umi':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='n_umi',
            hover_data=['tissue', 'cell_type_broad', 'timepoint', 'n_genes'],
            title=f"ZSCAPE Control Populations by UMI Count {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_continuous_scale='plasma'
        )
    
    # Update layout for dark theme with larger viewport
    fig.update_layout(
        scene=dict(
            bgcolor='#0e1117',
            xaxis=dict(
                backgroundcolor='#0e1117',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True,
                title_font=dict(color='#fafafa'),
                tickfont=dict(color='#fafafa')
            ),
            yaxis=dict(
                backgroundcolor='#0e1117',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True,
                title_font=dict(color='#fafafa'),
                tickfont=dict(color='#fafafa')
            ),
            zaxis=dict(
                backgroundcolor='#0e1117',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True,
                title_font=dict(color='#fafafa'),
                tickfont=dict(color='#fafafa')
            )
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        height=1500,  # Increased height significantly
        title_font=dict(size=18, color='#fafafa'),
        margin=dict(l=0, r=0, t=60, b=0)  # Minimize margins for more space
    )
    
    # Update traces for better visibility
    fig.update_traces(
        marker=dict(
            size=3,
            opacity=opacity,
            line=dict(width=0)
        )
    )
    
    return fig

def create_2d_summary_plots(df):
    """Create 2D summary plots for overview."""
    
    if len(df) == 0:
        return None, None
    
    # Sample if needed
    if len(df) > 20000:
        plot_df = df.sample(n=20000, random_state=42)
    else:
        plot_df = df
    
    # Tissue distribution plot
    fig1 = px.scatter(
        plot_df, x='x', y='y', 
        color='tissue',
        title="Control Populations by Tissue (2D Projection)",
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data=['cell_type_broad', 'timepoint']
    )
    
    # Timepoint distribution plot
    fig2 = px.scatter(
        plot_df, x='x', y='y',
        color='timepoint',
        title="Control Populations by Timepoint (2D Projection)",
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        color_continuous_scale='viridis',
        hover_data=['tissue', 'cell_type_broad']
    )
    
    # Apply dark theme with larger height
    for fig in [fig1, fig2]:
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font=dict(color='#fafafa'),
            height=1000  # Increased height
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_traces(marker=dict(size=2, opacity=0.6))
    
    return fig1, fig2

def display_control_metrics(df):
    """Display metrics about the control populations."""
    
    st.markdown("### 📊 Control Population Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🧬 Total Cells",
            value=f"{len(df):,}",
            help="Number of control cells after filtering"
        )
    
    with col2:
        st.metric(
            label="🏗️ Tissues",
            value=f"{df['tissue'].nunique()}",
            help="Number of unique tissue types"
        )
    
    with col3:
        st.metric(
            label="⏰ Timepoints", 
            value=f"{df['timepoint'].nunique()}",
            help="Number of developmental timepoints"
        )
    
    with col4:
        st.metric(
            label="🔬 Cell Types",
            value=f"{df['cell_type_broad'].nunique()}",
            help="Number of broad cell type categories"
        )
    
    # Quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mean_umi = df['n_umi'].mean()
        st.metric(
            label="📈 Mean UMIs",
            value=f"{mean_umi:.0f}",
            help="Average UMI count per cell"
        )
    
    with col2:
        mean_genes = df['n_genes'].mean()
        st.metric(
            label="🧮 Mean Genes",
            value=f"{mean_genes:.0f}",
            help="Average genes detected per cell"
        )
    
    with col3:
        if 'pct_mito' in df.columns:
            mean_mito = df['pct_mito'].mean()
            st.metric(
                label="⚡ Mean % Mito",
                value=f"{mean_mito:.1f}%",
                help="Average mitochondrial gene percentage"
            )

def create_distribution_plots(df):
    """Create distribution plots for quality metrics."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('UMI Count Distribution', 'Gene Count Distribution', 
                       'Tissue Distribution', 'Timepoint Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # UMI distribution
    fig.add_trace(
        go.Histogram(x=df['n_umi'], nbinsx=50, name='UMI Count', 
                    marker_color='#FF6B6B', opacity=0.7),
        row=1, col=1
    )
    
    # Gene distribution
    fig.add_trace(
        go.Histogram(x=df['n_genes'], nbinsx=50, name='Gene Count',
                    marker_color='#4ECDC4', opacity=0.7),
        row=1, col=2
    )
    
    # Tissue distribution
    tissue_counts = df['tissue'].value_counts().head(15)
    fig.add_trace(
        go.Bar(x=tissue_counts.index, y=tissue_counts.values, name='Tissue',
               marker_color='#45B7D1', opacity=0.7),
        row=2, col=1
    )
    
    # Timepoint distribution
    tp_counts = df['timepoint'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=tp_counts.index, y=tp_counts.values, name='Timepoint',
               marker_color='#96CEB4', opacity=0.7),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1500,  # Increased height
        showlegend=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        title_text="Control Population Distributions"
    )
    
    # Update all axes
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', tickangle=45, row=2, col=1)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def main():
    # Header
    st.markdown("""
    <h1 style='color: #fafafa; text-align: center; margin-bottom: 0;'>
        🧬 ZSCAPE Control Populations Explorer
    </h1>
    <p style='color: #aaaaaa; text-align: center; font-size: 18px; margin-top: 10px;'>
        Explore unperturbed zebrafish developmental atlas • 285k+ control cells
    </p>
    <hr style='border: 1px solid #333; margin: 20px 0;'>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading control population data..."):
        df = load_control_data()
        tissues, timepoints, cell_types_broad = get_filter_options(df)
    
    # Sidebar controls
    st.sidebar.markdown("<h2 style='color: #fafafa;'>🧬 Exploration Controls</h2>", unsafe_allow_html=True)
    
    # Multi-select filters
    tissue_options = ["All Tissues"] + tissues
    selected_tissues = st.sidebar.multiselect(
        "Select Tissues",
        options=tissue_options,
        default=["All Tissues"],
        help="Choose specific tissues to explore"
    )
    
    if "All Tissues" in selected_tissues or len(selected_tissues) == 0:
        selected_tissues = ["All Tissues"]
    
    timepoint_options = ["All Timepoints"] + [f"{tp} hpf" for tp in timepoints]
    selected_timepoints = st.sidebar.multiselect(
        "Select Timepoints",
        options=timepoint_options,
        default=["All Timepoints"],
        help="Choose developmental stages"
    )
    
    if "All Timepoints" in selected_timepoints or len(selected_timepoints) == 0:
        selected_timepoints = ["All Timepoints"]
    
    cell_type_options = ["All Cell Types"] + cell_types_broad
    selected_cell_types = st.sidebar.multiselect(
        "Select Cell Types",
        options=cell_type_options,
        default=["All Cell Types"],
        help="Choose broad cell type categories"
    )
    
    if "All Cell Types" in selected_cell_types or len(selected_cell_types) == 0:
        selected_cell_types = ["All Cell Types"]
    
    # Quality filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color: #fafafa;'>🔍 Quality Filters</h3>", unsafe_allow_html=True)
    
    umi_range = st.sidebar.slider(
        "UMI Count Range",
        min_value=int(df['n_umi'].min()),
        max_value=int(df['n_umi'].max()),
        value=(int(df['n_umi'].quantile(0.05)), int(df['n_umi'].quantile(0.95))),
        help="Filter cells by UMI count"
    )
    
    gene_range = st.sidebar.slider(
        "Gene Count Range", 
        min_value=int(df['n_genes'].min()),
        max_value=int(df['n_genes'].max()),
        value=(int(df['n_genes'].quantile(0.05)), int(df['n_genes'].quantile(0.95))),
        help="Filter cells by number of detected genes"
    )
    
    # Visualization controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color: #fafafa;'>🎨 Visualization</h3>", unsafe_allow_html=True)
    
    color_by = st.sidebar.selectbox(
        "Color Points By",
        options=['tissue', 'cell_type_broad', 'timepoint', 'n_umi'],
        index=0,
        help="Choose what to color the points by"
    )
    
    size_by = st.sidebar.selectbox(
        "Size Points By",
        options=['fixed', 'n_umi', 'n_genes'],
        index=0,
        help="Choose what determines point size"
    )
    
    opacity = st.sidebar.slider(
        "Point Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust point transparency"
    )
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color: #fafafa;'>📊 Dataset Info</h3>", unsafe_allow_html=True)
    
    st.sidebar.info(f"""
    **Source:** ZSCAPE ctrl-inj population
    
    **Total Control Cells:** {len(df):,}
    
    **Tissues:** {len(tissues)}
    
    **Timepoints:** {len(timepoints)}
    
    **Cell Types:** {len(cell_types_broad)}
    
    **Embedding:** Pre-computed 3D UMAP
    """)
    
    # Filter data
    filtered_df = filter_control_data(
        df, selected_tissues, selected_timepoints, selected_cell_types,
        umi_range, gene_range, max_points=50000
    )
    
    # Main content
    if len(filtered_df) == 0:
        st.warning("No cells match the selected filters. Please adjust your criteria.")
        return
    
    # Main visualization tabs - now full width
    tab1, tab2, tab3 = st.tabs(["3D Visualization", "2D Overview", "Distributions"])
    
    with tab1:
        st.markdown("### 🌌 3D Control Population Landscape")
        
        fig_3d = create_3d_scatter_plot(filtered_df, color_by=color_by, size_by=size_by, opacity=opacity)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown(f"""
        <div style='background-color: #262730; padding: 15px; border-radius: 5px; margin: 10px 0;'>
            <b>Current View:</b> {len(filtered_df):,} control cells<br>
            <b>Coloring:</b> {color_by.replace('_', ' ').title()}<br>
            <b>Point Size:</b> {size_by.replace('_', ' ').title()}
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 2D Population Overview")
        
        fig1, fig2 = create_2d_summary_plots(filtered_df)
        if fig1 and fig2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Quality and Distribution Analysis")
        
        dist_fig = create_distribution_plots(filtered_df)
        st.plotly_chart(dist_fig, use_container_width=True)
    
    # Metrics section moved below the visualization
    st.markdown("---")
    st.markdown("## 📊 Population Analysis & Metrics")
    
    # Display metrics below visualization
    display_control_metrics(filtered_df)
    
    # Data preview
    with st.expander("🔍 View Raw Data Preview"):
        display_cols = ['tissue', 'cell_type_broad', 'timepoint', 'n_umi', 'n_genes', 'x', 'y', 'z']
        st.dataframe(filtered_df[display_cols].head(100), height=300)

if __name__ == "__main__":
    main()