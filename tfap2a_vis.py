import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="ZSCAPE tfap2a Perturbation Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme CSS - same styling as controls
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
        padding: 2px;
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
        border-radius: 1px;
        color: #fafafa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_tfap2a_data():
    """Load prepared tfap2a perturbation data."""
    data_path = Path("demo_data/zscape_tfap2a_viz.csv")
    
    if not data_path.exists():
        st.error(f"""
        Perturbation data not found: {data_path}
        
        Please run the data preparation first:
        ```python
        python tfap2a_prep.py
        ```
        """)
        st.stop()
    
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def get_filter_options(df):
    """Get available options for filtering."""
    tissues = sorted(df['tissue'].astype(str).unique())
    timepoints = sorted(df['timepoint'].astype(str).unique())
    cell_types_broad = sorted(df['cell_type_broad'].astype(str).unique())
    perturbations = sorted(df['perturbation_type'].astype(str).unique())
    
    return tissues, timepoints, cell_types_broad, perturbations

@st.cache_data
def filter_perturbation_data(df, selected_tissues, selected_timepoints, selected_cell_types,
                           selected_perturbations, max_points=50000):
    """Filter perturbation data based on selections."""
    
    # Apply filters
    filtered_df = df.copy()
    
    if "All Tissues" not in selected_tissues:
        filtered_df = filtered_df[filtered_df['tissue'].isin(selected_tissues)]
    
    if "All Timepoints" not in selected_timepoints:
        timepoint_values = []
        for tp in selected_timepoints:
            if 'hpf' in str(tp):
                timepoint_values.append(str(tp).split()[0])
            else:
                timepoint_values.append(str(tp))
        filtered_df = filtered_df[filtered_df['timepoint'].astype(str).isin(timepoint_values)]
    
    if "All Cell Types" not in selected_cell_types:
        filtered_df = filtered_df[filtered_df['cell_type_broad'].isin(selected_cell_types)]
    
    if "All Perturbations" not in selected_perturbations:
        filtered_df = filtered_df[filtered_df['perturbation_type'].isin(selected_perturbations)]
    
    # Sample for performance if needed
    if len(filtered_df) > max_points:
        filtered_df = filtered_df.sample(n=max_points, random_state=42)
        st.sidebar.warning(f"⚠️ Showing random sample of {max_points:,} cells for performance")
    
    return filtered_df

def create_3d_scatter_plot(df, color_by='perturbation_type', opacity=0.7):
    """Create 3D scatter plot of perturbation populations."""
    
    if len(df) == 0:
        return None
    
    # Sample further if still too many points for 3D
    if len(df) > 30000:
        plot_df = df.sample(n=30000, random_state=42)
        sample_note = f"(showing {30000:,} of {len(df):,} cells)"
    else:
        plot_df = df
        sample_note = f"({len(df):,} cells)"
    
    # Create 3D scatter plot
    if color_by == 'perturbation_type':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='perturbation_type',
            hover_data=['tissue', 'cell_type_broad', 'timepoint'],
            title=f"tfap2a Perturbation Populations in 3D Space {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
    elif color_by == 'tissue':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='tissue',
            hover_data=['perturbation_type', 'cell_type_broad', 'timepoint'],
            title=f"tfap2a Perturbation Populations by Tissue {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    elif color_by == 'cell_type_broad':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='cell_type_broad',
            hover_data=['perturbation_type', 'tissue', 'timepoint'],
            title=f"tfap2a Perturbation Populations by Cell Type {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
    elif color_by == 'timepoint':
        fig = px.scatter_3d(
            plot_df,
            x='x', y='y', z='z',
            color='timepoint',
            hover_data=['perturbation_type', 'tissue', 'cell_type_broad'],
            title=f"tfap2a Perturbation Populations by Timepoint {sample_note}",
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'z': 'UMAP 3'},
            color_continuous_scale='viridis'
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
    
    # Perturbation distribution plot
    fig1 = px.scatter(
        plot_df, x='x', y='y', 
        color='perturbation_type',
        title="Perturbation Populations by Type (2D Projection)",
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=['tissue', 'cell_type_broad', 'timepoint']
    )
    
    # Tissue distribution plot
    fig2 = px.scatter(
        plot_df, x='x', y='y',
        color='tissue',
        title="Perturbation Populations by Tissue (2D Projection)",
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        hover_data=['perturbation_type', 'cell_type_broad', 'timepoint']
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

def display_perturbation_metrics(df):
    """Display metrics about the perturbation populations."""
    
    st.markdown("### 📊 Perturbation Population Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🔬 Total Cells",
            value=f"{len(df):,}",
            help="Number of perturbed cells after filtering"
        )
    
    with col2:
        st.metric(
            label="⚗️ Perturbations",
            value=f"{df['perturbation_type'].nunique()}",
            help="Number of different perturbation types"
        )
    
    with col3:
        st.metric(
            label="🏗️ Tissues",
            value=f"{df['tissue'].nunique()}",
            help="Number of unique tissue types"
        )
    
    with col4:
        st.metric(
            label="🔬 Cell Types",
            value=f"{df['cell_type_broad'].nunique()}",
            help="Number of broad cell type categories"
        )
    
    # Perturbation-specific metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="⏰ Timepoints", 
            value=f"{df['timepoint'].nunique()}",
            help="Number of developmental timepoints"
        )
    
    with col2:
        # Show most common perturbation
        most_common_pert = df['perturbation_type'].mode().iloc[0] if len(df) > 0 else "N/A"
        st.metric(
            label="🎯 Top Perturbation",
            value=most_common_pert,
            help="Most abundant perturbation type"
        )
    
    with col3:
        # Show most common tissue
        most_common_tissue = df['tissue'].mode().iloc[0] if len(df) > 0 else "N/A"
        st.metric(
            label="🧬 Top Tissue",
            value=most_common_tissue,
            help="Most abundant tissue type"
        )

def create_distribution_plots(df):
    """Create distribution plots for perturbation analysis."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Perturbation Type Distribution', 'Tissue Distribution', 
                       'Cell Type Distribution', 'Timepoint Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Perturbation distribution
    pert_counts = df['perturbation_type'].value_counts()
    fig.add_trace(
        go.Bar(x=pert_counts.index, y=pert_counts.values, name='Perturbation',
               marker_color='#4ECDC4', opacity=0.7),
        row=1, col=1
    )
    
    # Tissue distribution
    tissue_counts = df['tissue'].value_counts().head(15)
    fig.add_trace(
        go.Bar(x=tissue_counts.index, y=tissue_counts.values, name='Tissue',
               marker_color='#45B7D1', opacity=0.7),
        row=1, col=2
    )
    
    # Cell type distribution
    cell_counts = df['cell_type_broad'].value_counts().head(15)
    fig.add_trace(
        go.Bar(x=cell_counts.index, y=cell_counts.values, name='Cell Type',
               marker_color='#96CEB4', opacity=0.7),
        row=2, col=1
    )
    
    # Timepoint distribution
    tp_counts = df['timepoint'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(x=tp_counts.index, y=tp_counts.values, name='Timepoint',
               marker_color='#FECA57', opacity=0.7),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1500,  # Increased height
        showlegend=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        title_text="Perturbation Population Distributions"
    )
    
    # Update all axes
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', tickangle=45)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def main():
    # Header
    st.markdown("""
    <h1 style='color: #fafafa; text-align: center; margin-bottom: 0;'>
        🔬 ZSCAPE tfap2a Perturbation Explorer
    </h1>
    <p style='color: #aaaaaa; text-align: center; font-size: 18px; margin-top: 2px;'>
        Explore tfap2a perturbed zebrafish developmental cells • Single & Combo perturbations
    </p>
    <hr style='border: 1px solid #333; margin: 2px 0;'>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading perturbation population data..."):
        df = load_tfap2a_data()
        tissues, timepoints, cell_types_broad, perturbations = get_filter_options(df)
    
    # Sidebar controls
    st.sidebar.markdown("<h2 style='color: #fafafa;'>🔬 Perturbation Controls</h2>", unsafe_allow_html=True)
    
    # Multi-select filters
    perturbation_options = ["All Perturbations"] + perturbations
    selected_perturbations = st.sidebar.multiselect(
        "Select Perturbations",
        options=perturbation_options,
        default=["All Perturbations"],
        help="Choose specific perturbation types to explore"
    )
    
    if "All Perturbations" in selected_perturbations or len(selected_perturbations) == 0:
        selected_perturbations = ["All Perturbations"]
    
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
    
    # Visualization controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color: #fafafa;'>🎨 Visualization</h3>", unsafe_allow_html=True)
    
    color_by = st.sidebar.selectbox(
        "Color Points By",
        options=['perturbation_type', 'tissue', 'cell_type_broad', 'timepoint'],
        index=0,
        help="Choose what to color the points by"
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
    **Source:** ZSCAPE tfap2a perturbations
    
    **Total Perturbed Cells:** {len(df):,}
    
    **Perturbation Types:** {len(perturbations)}
    
    **Tissues:** {len(tissues)}
    
    **Timepoints:** {len(timepoints)}
    
    **Cell Types:** {len(cell_types_broad)}
    
    **Embedding:** Pre-computed 3D UMAP
    """)
    
    # Filter data
    filtered_df = filter_perturbation_data(
        df, selected_tissues, selected_timepoints, selected_cell_types,
        selected_perturbations, max_points=50000
    )
    
    # Main content
    if len(filtered_df) == 0:
        st.warning("No cells match the selected filters. Please adjust your criteria.")
        return
    
    # Main visualization tabs - now full width
    tab1, tab2, tab3 = st.tabs(["3D Visualization", "2D Overview", "Distributions"])
    
    with tab1:
        st.markdown("### 🌌 3D Perturbation Population Landscape")
        
        fig_3d = create_3d_scatter_plot(filtered_df, color_by=color_by, opacity=opacity)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown(f"""
        <div style='background-color: #262730; padding: 15px; border-radius: 5px; margin: 2px 0;'>
            <b>Current View:</b> {len(filtered_df):,} perturbed cells<br>
            <b>Coloring:</b> {color_by.replace('_', ' ').title()}<br>
            <b>Perturbations:</b> {', '.join(filtered_df['perturbation_type'].unique()[:3])}{'...' if len(filtered_df['perturbation_type'].unique()) > 3 else ''}
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
        st.markdown("### 📈 Distribution Analysis")
        
        dist_fig = create_distribution_plots(filtered_df)
        st.plotly_chart(dist_fig, use_container_width=True)
    
    # Metrics section moved below the visualization
    st.markdown("---")
    st.markdown("## 📊 Perturbation Analysis & Metrics")
    
    # Display metrics below visualization
    display_perturbation_metrics(filtered_df)
    
    # Data preview
    with st.expander("🔍 View Raw Data Preview"):
        display_cols = ['perturbation_type', 'tissue', 'cell_type_broad', 'timepoint', 'x', 'y', 'z']
        st.dataframe(filtered_df[display_cols].head(100), height=300)

if __name__ == "__main__":
    main()