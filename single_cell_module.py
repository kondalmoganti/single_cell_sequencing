"""
Single-Cell RNA-seq Analysis Module for OmicsInsight
Integrates single-cell sequencing analysis capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import tempfile
from pathlib import Path
import io

# Lazy imports for heavy dependencies
try:
    import scanpy as sc
    import anndata as ad
    SCANPY_OK = True
except ImportError:
    SCANPY_OK = False

try:
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ========================================================================================
# SINGLE-CELL DATA LOADING
# ========================================================================================

@st.cache_data(show_spinner=False)
def read_h5ad_from_bytes(raw: bytes):
    """Read h5ad file from bytes"""
    if not SCANPY_OK:
        st.error("scanpy and anndata are required for single-cell analysis. Install with: pip install scanpy anndata")
        return None
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        adata = ad.read_h5ad(tmp.name)
    return adata


@st.cache_data(show_spinner=False)
def read_10x_h5(raw: bytes):
    """Read 10x HDF5 file from bytes"""
    if not SCANPY_OK:
        st.error("scanpy is required. Install with: pip install scanpy")
        return None
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        adata = sc.read_10x_h5(tmp.name)
    return adata


def _apply_demo_cap(adata, cap: int = 5000, seed: int = 0):
    """Downsample cells to cap if enabled and adata.n_obs > cap"""
    if adata.n_obs <= int(cap):
        return adata, False
    rs = np.random.RandomState(int(seed))
    idx = rs.choice(adata.n_obs, int(cap), replace=False)
    adata2 = adata[idx, :].copy()
    return adata2, True


def _umap_scatter(adata, color_key=None, title="UMAP"):
    """Create UMAP scatter plot"""
    if not PLOTLY_OK or "X_umap" not in adata.obsm:
        return None
    
    umap = adata.obsm["X_umap"]
    df = pd.DataFrame(
        {"UMAP1": umap[:, 0], "UMAP2": umap[:, 1]},
        index=adata.obs_names
    )
    
    if color_key and color_key in adata.obs.columns:
        col = adata.obs[color_key].astype("object")
        col = col.where(pd.notna(col), other="unassigned")
        df["color"] = col.values
        
        cats = [c for c in pd.unique(df["color"]) if c != "unassigned"]
        cats = sorted(cats) + (["unassigned"] if "unassigned" in df["color"].values else [])
        
        color_map = {c: None for c in cats}
        if "unassigned" in cats:
            color_map["unassigned"] = "#9e9e9e"
        
        fig = px.scatter(
            df, x="UMAP1", y="UMAP2",
            color="color",
            category_orders={"color": cats},
            color_discrete_map=color_map,
            title=title
        )
    else:
        fig = px.scatter(df, x="UMAP1", y="UMAP2", title=title)
    
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ========================================================================================
# SINGLE-CELL QC DASHBOARD
# ========================================================================================

def show_single_cell_qc_dashboard(adata):
    """Show QC dashboard for single-cell data"""
    if not SCANPY_OK:
        st.error("scanpy is required for single-cell QC. Install with: pip install scanpy")
        return
    
    st.subheader("🔬 Single-Cell Quality Control")
    
    if adata is None:
        st.warning("⚠️ No single-cell data loaded. Please upload .h5ad or 10x data first.")
        return
    
    st.info(f"**Loaded:** {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Compute QC metrics if not present
    if "total_counts" not in adata.obs.columns or "n_genes_by_counts" not in adata.obs.columns:
        st.markdown("### Compute QC Metrics")
        gene_prefix = st.text_input(
            "Mitochondrial gene prefix (human: 'MT-', mouse: 'mt-')",
            value="MT-",
            key="sc_mt_prefix"
        )
        
        # Detect mitochondrial genes
        var_symbols = None
        for cand in ["gene_symbols", "gene_symbol", "features", "name", "symbol"]:
            if cand in adata.var.columns:
                var_symbols = adata.var[cand].astype(str)
                break
        if var_symbols is None:
            var_symbols = adata.var_names.astype(str)
        
        mt_genes_mask = var_symbols.str.upper().str.startswith(gene_prefix.upper())
        adata.var["mt"] = np.asarray(mt_genes_mask).astype(bool)
        
        if st.button("Compute QC Metrics", key="sc_compute_qc"):
            try:
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
                st.session_state.sc_adata = adata
                st.success("✅ QC metrics computed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"QC computation failed: {e}")
        return
    
    # Show QC plots
    st.markdown("### QC Metrics Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cells", f"{adata.n_obs:,}")
    with col2:
        st.metric("Total Genes", f"{adata.n_vars:,}")
    with col3:
        if "pct_counts_mt" in adata.obs.columns:
            mean_mt = adata.obs["pct_counts_mt"].mean()
            st.metric("Mean MT %", f"{mean_mt:.2f}%")
    
    # QC distributions
    if PLOTLY_OK:
        col1, col2 = st.columns(2)
        
        with col1:
            if "total_counts" in adata.obs.columns:
                fig = px.histogram(
                    x=adata.obs["total_counts"],
                    nbins=50,
                    title="Total Counts Distribution",
                    labels={"x": "Total Counts", "y": "Number of Cells"}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "n_genes_by_counts" in adata.obs.columns:
                fig = px.histogram(
                    x=adata.obs["n_genes_by_counts"],
                    nbins=50,
                    title="Genes per Cell Distribution",
                    labels={"x": "Number of Genes", "y": "Number of Cells"}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if "pct_counts_mt" in adata.obs.columns:
            fig = px.histogram(
                x=adata.obs["pct_counts_mt"],
                nbins=50,
                title="Mitochondrial % Distribution",
                labels={"x": "MT %", "y": "Number of Cells"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Filtering parameters
    st.markdown("### Filter Cells and Genes")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_genes_min = st.number_input("Min genes per cell", value=200, step=50, key="sc_min_genes")
    with col2:
        n_genes_max = st.number_input("Max genes per cell", value=6000, step=500, key="sc_max_genes")
    with col3:
        mt_max = st.slider("Max mito %", min_value=0, max_value=100, value=20, key="sc_max_mt")
    
    if st.button("Apply Filters", key="sc_apply_filters"):
        before = (adata.n_obs, adata.n_vars)
        
        sc.pp.filter_cells(adata, min_genes=int(n_genes_min))
        adata = adata[adata.obs["n_genes_by_counts"] <= int(n_genes_max)].copy()
        if "pct_counts_mt" in adata.obs.columns:
            adata = adata[adata.obs["pct_counts_mt"] <= float(mt_max)].copy()
        sc.pp.filter_genes(adata, min_cells=3)
        
        after = (adata.n_obs, adata.n_vars)
        st.info(f"Filtered: {before[0]} cells → {after[0]} cells | {before[1]} genes → {after[1]} genes")
        st.session_state.sc_adata = adata
        st.rerun()
    
    # QC summary table
    qc_cols = [c for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt"] if c in adata.obs.columns]
    if qc_cols:
        st.markdown("### QC Metrics Table (Sample)")
        st.dataframe(adata.obs[qc_cols].head(20), use_container_width=True)


# ========================================================================================
# SINGLE-CELL NORMALIZATION
# ========================================================================================

def show_single_cell_normalization(adata):
    """Show normalization interface for single-cell data"""
    if not SCANPY_OK:
        st.error("scanpy is required. Install with: pip install scanpy")
        return
    
    if adata is None:
        st.warning("⚠️ No single-cell data loaded.")
        return
    
    st.subheader("⚙️ Normalization & Highly Variable Genes")
    
    # Normalization
    st.markdown("### Normalization")
    method = st.selectbox(
        "Normalization Method",
        ["Normalize Total + Log1p", "SCVI (variance-stabilizing)"],
        key="sc_norm_method"
    )
    
    if method == "Normalize Total + Log1p":
        target_sum = st.number_input(
            "Target sum per cell",
            value=1e4, step=1e3, format="%.0f", key="sc_target_sum"
        )
        if st.button("Run Normalization", key="sc_run_norm"):
            with st.spinner("Normalizing..."):
                sc.pp.normalize_total(adata, target_sum=float(target_sum))
                sc.pp.log1p(adata)
            st.session_state.sc_adata = adata
            st.success("✅ Normalized and log1p-transformed!")
            st.rerun()
    else:
        st.info("💡 SCVI normalization requires scvi-tools. Install with: pip install scvi-tools")
    
    # HVG detection
    st.markdown("### Highly Variable Genes (HVGs)")
    
    if "highly_variable" not in adata.var.columns:
        flavor = st.selectbox(
            "HVG Detection Method",
            ["cellranger", "seurat", "seurat_v3"],
            key="sc_hvg_flavor"
        )
        
        n_top_genes = st.number_input(
            "Number of top variable genes",
            min_value=500,
            max_value=5000,
            value=2000,
            step=100,
            key="sc_n_top_genes"
        )
        
        if st.button("Find HVGs", key="sc_find_hvgs"):
            with st.spinner("Finding highly variable genes..."):
                sc.pp.highly_variable_genes(adata, n_top_genes=int(n_top_genes), flavor=flavor)
            st.session_state.sc_adata = adata
            st.success(f"✅ Found {adata.var['highly_variable'].sum()} highly variable genes!")
            st.rerun()
    else:
        n_hvg = adata.var["highly_variable"].sum()
        st.success(f"✅ Already computed: {n_hvg} highly variable genes")
        
        # Show HVG table
        if st.checkbox("Show HVG table", key="sc_show_hvg"):
            hvgs = adata.var[adata.var["highly_variable"]].sort_values("dispersions_norm", ascending=False)
            st.dataframe(hvgs.head(50), use_container_width=True)


# ========================================================================================
# SINGLE-CELL EMBEDDING & CLUSTERING
# ========================================================================================

def show_single_cell_embedding_clustering(adata):
    """Show embedding and clustering interface"""
    if not SCANPY_OK:
        st.error("scanpy is required. Install with: pip install scanpy")
        return
    
    if adata is None:
        st.warning("⚠️ No single-cell data loaded.")
        return
    
    st.subheader("🌀 Dimensionality Reduction & Clustering")
    
    # Embedding controls
    col1, col2 = st.columns(2)
    with col1:
        n_pcs = st.slider("Number of PCs", 10, 100, 50, key="sc_n_pcs")
    with col2:
        neighbors_k = st.slider("Neighbors k", 5, 50, 15, key="sc_neighbors_k")
    
    use_hvg = st.checkbox("Use only HVGs for PCA", value=True, key="sc_use_hvg")
    
    if st.button("Run PCA/UMAP", key="sc_run_embedding"):
        with st.spinner("Computing PCA and UMAP..."):
            adata_use = adata
            if use_hvg and "highly_variable" in adata.var.columns:
                adata_use = adata[:, adata.var["highly_variable"]].copy()
            
            sc.pp.scale(adata_use, max_value=10)
            sc.tl.pca(adata_use, n_comps=int(n_pcs))
            sc.pp.neighbors(adata_use, n_neighbors=int(neighbors_k), n_pcs=int(n_pcs))
            sc.tl.umap(adata_use)
            
            # Copy results back
            for k in ["X_pca", "X_umap"]:
                if k in adata_use.obsm:
                    adata.obsm[k] = adata_use.obsm[k]
            for k in ["distances", "connectivities"]:
                if k in adata_use.obsp:
                    adata.obsp[k] = adata_use.obsp[k]
        
        st.session_state.sc_adata = adata
        st.success("✅ Computed PCA and UMAP!")
        st.rerun()
    
    # Clustering
    st.markdown("### Clustering")
    
    algo = st.selectbox(
        "Clustering Algorithm",
        ["Leiden", "Louvain", "KMeans"],
        index=0,
        key="sc_cluster_algo"
    )
    
    resolution = st.slider(
        "Resolution (for Leiden/Louvain)",
        0.1, 2.0, 0.5, 0.1,
        key="sc_resolution"
    )
    
    k_kmeans = st.slider("K (for KMeans)", 2, 50, 10, key="sc_k_kmeans")
    
    if st.button("Run Clustering", key="sc_run_clustering"):
        with st.spinner("Clustering..."):
            try:
                if algo == "Leiden":
                    sc.tl.leiden(adata, resolution=float(resolution))
                elif algo == "Louvain":
                    sc.tl.louvain(adata, resolution=float(resolution))
                else:  # KMeans
                    from sklearn.cluster import KMeans
                    if "X_pca" not in adata.obsm:
                        sc.pp.scale(adata, max_value=10)
                        sc.tl.pca(adata, n_comps=50)
                    X = adata.obsm["X_pca"]
                    labels = KMeans(n_clusters=int(k_kmeans), n_init=10, random_state=0).fit_predict(X)
                    adata.obs["kmeans"] = pd.Categorical(labels.astype(str))
                st.session_state.sc_adata = adata
                st.success(f"✅ {algo} clustering complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Clustering failed: {e}")
    
    # UMAP visualization
    if "X_umap" in adata.obsm:
        st.markdown("### UMAP Visualization")
        
        # Color by options
        color_options = ["(none)"]
        cluster_keys = [c for c in ["leiden", "louvain", "kmeans"] if c in adata.obs.columns]
        color_options.extend(cluster_keys)
        
        for c in adata.obs.columns:
            if c not in cluster_keys:
                s = adata.obs[c]
                if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
                    if s.nunique() <= 50:
                        color_options.append(c)
        
        color_by = st.selectbox(
            "Color UMAP by",
            options=color_options,
            index=1 if cluster_keys else 0,
            key="sc_umap_color"
        )
        
        color_key = None if color_by == "(none)" else color_by
        
        fig = _umap_scatter(adata, color_key=color_key)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        if color_key in cluster_keys:
            st.dataframe(
                adata.obs[color_key].value_counts().rename_axis("cluster").reset_index(name="n"),
                use_container_width=True
            )
    else:
        st.info("💡 Run PCA/UMAP first to visualize")


# ========================================================================================
# SINGLE-CELL MARKER ANALYSIS
# ========================================================================================

def show_single_cell_marker_analysis(adata):
    """Show marker gene analysis interface"""
    if not SCANPY_OK:
        st.error("scanpy is required. Install with: pip install scanpy")
        return
    
    if adata is None:
        st.warning("⚠️ No single-cell data loaded.")
        return
    
    st.subheader("🔬 Marker Gene Analysis")
    
    # Group selection
    candidate_groupbys = []
    for c in adata.obs.columns:
        s = adata.obs[c]
        if pd.api.types.is_categorical_dtype(s):
            if 2 <= len(s.cat.categories) <= 50:
                candidate_groupbys.append(c)
        elif s.dtype == object:
            nunq = s.nunique(dropna=True)
            if 2 <= nunq <= 50:
                candidate_groupbys.append(c)
    
    # Prioritize cluster columns
    for k in ["leiden", "louvain", "kmeans"]:
        if k in candidate_groupbys:
            candidate_groupbys.remove(k)
            candidate_groupbys.insert(0, k)
    
    if not candidate_groupbys:
        st.warning("No suitable grouping columns found. Run clustering first.")
        return
    
    groupby = st.selectbox("Group by", candidate_groupbys, key="sc_marker_groupby")
    
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        adata.obs[groupby] = adata.obs[groupby].astype("category")
    
    # Filter small groups
    min_cells = st.number_input("Minimum cells per group", 2, 100, 5, key="sc_min_cells")
    counts = adata.obs[groupby].value_counts()
    valid_groups = counts[counts >= int(min_cells)].index.tolist()
    
    if len(valid_groups) < 2:
        st.error("Not enough groups after filtering. Lower min_cells or adjust clustering.")
        return
    
    mask = adata.obs[groupby].isin(valid_groups)
    adata_use = adata[mask].copy()
    adata_use.obs[groupby] = adata_use.obs[groupby].cat.remove_unused_categories()
    
    # DE parameters
    cats = list(adata_use.obs[groupby].cat.categories)
    ref_choice = st.selectbox("Reference group", ["rest"] + cats, key="sc_de_reference")
    reference = "rest" if ref_choice == "rest" else ref_choice
    
    method = st.selectbox(
        "DE Test Method",
        ["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
        index=0,
        key="sc_de_method"
    )
    
    if st.button("Run Marker Analysis", key="sc_run_markers"):
        with st.spinner("Finding marker genes..."):
            try:
                sc.tl.rank_genes_groups(
                    adata_use,
                    groupby=groupby,
                    method=method,
                    reference=reference,
                )
                
                # Get results
                df = sc.get.rank_genes_groups_df(adata_use, group=None)
                
                st.success(f"✅ Found markers for {len(cats)} groups!")
                st.dataframe(df.head(100), use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Marker Results (CSV)",
                    csv,
                    file_name=f"marker_genes_{groupby}_{method}.csv",
                    mime="text/csv",
                    key="sc_download_markers"
                )
                
                # Store results
                adata.uns["rank_genes_groups"] = adata_use.uns["rank_genes_groups"]
                st.session_state.sc_adata = adata
                
            except Exception as e:
                st.error(f"Marker analysis failed: {e}")
    
    # Show existing results
    if "rank_genes_groups" in adata.uns:
        st.markdown("### Previous Results")
        try:
            df = sc.get.rank_genes_groups_df(adata, group=None)
            st.dataframe(df.head(100), use_container_width=True)
        except Exception:
            pass


# ========================================================================================
# MAIN SINGLE-CELL TAB INTERFACE
# ========================================================================================

def show_single_cell_data_quality_tab():
    """Show data quality tab for single-cell"""
    st.markdown("### 📊 Single-Cell Data Quality & Preparation")
    
    # Initialize single-cell data in session state
    if "sc_adata" not in st.session_state:
        st.session_state.sc_adata = None
    
    # Data loading
    st.markdown("#### 📁 Load Single-Cell Data")
    
    load_method = st.radio(
        "Load from:",
        [".h5ad File", "10x HDF5 (.h5)", "Expression Matrix (CSV)"],
        horizontal=True,
        key="sc_load_method"
    )
    
    if load_method == ".h5ad File":
        h5ad_file = st.file_uploader("Upload .h5ad file", type=["h5ad"], key="sc_h5ad_upload")
        if h5ad_file is not None:
            try:
                raw = h5ad_file.read()
                with st.spinner("Loading .h5ad file..."):
                    adata = read_h5ad_from_bytes(raw)
                if adata:
                    st.success(f"✅ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
                    st.session_state.sc_adata = adata
            except Exception as e:
                st.error(f"Error loading .h5ad: {e}")
    
    elif load_method == "10x HDF5 (.h5)":
        h5_file = st.file_uploader("Upload 10x HDF5 file", type=["h5"], key="sc_h5_upload")
        if h5_file is not None:
            try:
                raw = h5_file.read()
                with st.spinner("Loading 10x HDF5 file..."):
                    adata = read_10x_h5(raw)
                    # Add basic QC
                    X = adata.X
                    try:
                        total = np.asarray(X.sum(axis=1)).ravel()
                        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
                    except Exception:
                        total = X.sum(axis=1)
                        n_genes = (X > 0).sum(axis=1)
                    adata.obs["total_counts"] = total
                    adata.obs["n_genes_by_counts"] = n_genes
                if adata:
                    st.success(f"✅ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
                    st.session_state.sc_adata = adata
            except Exception as e:
                st.error(f"Error loading 10x HDF5: {e}")
    
    else:  # Expression Matrix
        st.info("💡 Upload a CSV expression matrix (genes × cells or cells × genes). Standard expression matrices work too.")
        # Can use regular expression matrix loading from main app
    
    st.markdown("---")
    
    # Show QC if data loaded
    if st.session_state.sc_adata is not None:
        show_single_cell_qc_dashboard(st.session_state.sc_adata)
        
        # Normalization
        with st.expander("⚙️ Normalization & HVGs", expanded=False):
            show_single_cell_normalization(st.session_state.sc_adata)
    else:
        st.info("💡 Upload single-cell data to begin analysis")


def show_single_cell_explore_tab():
    """Show exploration tab for single-cell"""
    if "sc_adata" not in st.session_state or st.session_state.sc_adata is None:
        st.warning("⚠️ Please load single-cell data in the Data Quality tab first.")
        return
    
    st.markdown("### 🔍 Single-Cell Exploration")
    
    subtab_choice = st.radio(
        "Select analysis:",
        ["🌀 Embedding & Clustering", "📊 UMAP Visualization"],
        horizontal=True,
        key="sc_explore_subtab"
    )
    
    st.markdown("---")
    
    if subtab_choice == "🌀 Embedding & Clustering":
        show_single_cell_embedding_clustering(st.session_state.sc_adata)
    else:
        # UMAP visualization
        if "X_umap" in st.session_state.sc_adata.obsm:
            color_options = ["(none)"]
            cluster_keys = [c for c in ["leiden", "louvain", "kmeans"] if c in st.session_state.sc_adata.obs.columns]
            color_options.extend(cluster_keys)
            
            color_by = st.selectbox(
                "Color UMAP by",
                options=color_options,
                index=1 if cluster_keys else 0,
                key="sc_umap_view_color"
            )
            
            color_key = None if color_by == "(none)" else color_by
            fig = _umap_scatter(st.session_state.sc_adata, color_key=color_key)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("💡 Run embedding and clustering first to visualize UMAP")


def show_single_cell_marker_analysis_tab():
    """Show marker analysis tab for single-cell"""
    if "sc_adata" not in st.session_state or st.session_state.sc_adata is None:
        st.warning("⚠️ Please load single-cell data and run clustering first.")
        return
    
    show_single_cell_marker_analysis(st.session_state.sc_adata)
