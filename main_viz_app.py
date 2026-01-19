import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import io
from volcano_plot_module import show_volcano_plot
from visualization import show_visualizations
from overlap_plots import show_overlap_tools
from normalization import normalize_df, select_normalization_method
from multi_panel_figure import show_multi_panel_figure
from gene_list_manager import show_gene_list_manager_sidebar
from qc_dashboard import show_qc_dashboard
from counts_de_module import show_counts_de
from export_ui import show_export_manager

# NEW: Import validation utilities
from utils.file_io import read_delimited_file
from utils.validation import (
    FileValidator,
    DataValidator,
    ValidationError
)

# ------------------------ Display Helpers ------------------------
def _format_pvals_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format very small p-values for display so they don't appear as 0 in Streamlit tables.
    Display-only: do not use the returned dataframe for downstream calculations.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()

    p_cols = [c for c in out.columns if c.lower() in {"pvalue","p_value","pval","p_val"}]
    q_cols = [c for c in out.columns if c.lower() in {"padj","adj_p","adj_pvalue","qvalue","q_value","fdr"}]

    # Add -log10 columns if possible (numeric)
    for c in p_cols:
        try:
            x = pd.to_numeric(out[c], errors="coerce").clip(lower=1e-300)
            out[f"-log10({c})"] = (-np.log10(x)).round(3)
        except Exception:
            pass
    for c in q_cols:
        try:
            x = pd.to_numeric(out[c], errors="coerce").clip(lower=1e-300)
            out[f"-log10({c})"] = (-np.log10(x)).round(3)
        except Exception:
            pass

    # Format p/q columns as scientific notation strings (display only)
    for c in p_cols + q_cols:
        try:
            x = pd.to_numeric(out[c], errors="coerce")
            out[c] = x.map(lambda v: f"{v:.2e}" if pd.notna(v) else "")
        except Exception:
            continue

    return out



# ------------------------ Page Setup ------------------------
st.set_page_config(page_title="🧬 OmicsInsight | Interactive Omics Data Analysis & Visualization", layout="wide")

# ------------------------ Initialize Session State ------------------------
if "df" not in st.session_state:
    st.session_state["df"] = None
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None
if "sample_metadata" not in st.session_state:
    st.session_state["sample_metadata"] = None
if "sample_metadata_name" not in st.session_state:
    st.session_state["sample_metadata_name"] = None
if "color_theme" not in st.session_state:
    from utils.color_themes import get_recommended_theme
    st.session_state["color_theme"] = get_recommended_theme("general")
if "volcano_theme" not in st.session_state:
    st.session_state["volcano_theme"] = "Default"


# ------------------------ File Loader ------------------------
@st.cache_data
def load_uploaded_data(file_obj, file_name):
    """Load CSV/TSV/TXT/Excel robustly from Streamlit UploadedFile (safe across reruns)."""
    if file_obj is None:
        return pd.DataFrame()

    try:
        raw = file_obj.getvalue()
    except Exception:
        try:
            file_obj.seek(0)
        except Exception:
            pass
        raw = file_obj.read()

    if not raw:
        return pd.DataFrame()

    if file_name.endswith((".csv", ".tsv", ".txt")):
        for sep in [",", "	", ";"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, comment="#")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(raw), comment="#")

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("Unsupported file type. Upload a CSV/TSV/TXT or Excel file.")


@st.cache_data
def load_metadata(file_obj, file_name):
    """Load sample metadata (CSV/TSV/TXT/Excel) robustly."""
    if file_obj is None:
        return pd.DataFrame()

    try:
        raw = file_obj.getvalue()
    except Exception:
        try:
            file_obj.seek(0)
        except Exception:
            pass
        raw = file_obj.read()

    if not raw:
        return pd.DataFrame()

    if file_name.endswith((".csv", ".tsv", ".txt")):
        for sep in [",", "	", ";"]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, comment="#")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
        return pd.read_csv(io.BytesIO(raw), comment="#")

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    raise ValueError("Unsupported metadata format.")


# ========================================================================================
# HEADER & TITLE
# ========================================================================================
st.title("🧬 OmicsInsight | Interactive Omics Data Analysis & Visualization")
st.caption("Interactive analysis and publication-ready figures for bioinformatics workflows")
st.markdown("---")

# ========================================================================================
# GLOBAL DATA UPLOADER (Collapsible)
# ========================================================================================
with st.expander("📁 **SHARED DATA** (used by QC, Visualization, Volcano, Gene Expression)", expanded=True):
    st.caption("Upload normalized expression data and optional metadata. This will be available across multiple sections.")
    
    # Example datasets section
    with st.expander("📚 Load Example Datasets", expanded=False):
        from example_datasets import show_example_loader
        show_example_loader()
    
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        uploaded_file_global = st.file_uploader(
            "📊 Dataset (Expression Matrix / DE Results)",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            key="global_dataset_upload",
            help="Upload normalized expression matrix or differential expression results"
        )

        if uploaded_file_global is not None:
            try:
                # Validate file
                FileValidator.validate_upload(uploaded_file_global)

                # Read file
                df = read_delimited_file(uploaded_file_global, validate=False)

                # Validate data format
                is_valid, error, suggestion = DataValidator.validate_expression_matrix(
                    df, min_rows=2, min_cols=2, min_numeric_cols=1
                )

                if not is_valid:
                    st.error(f"❌ {error}")
                    if suggestion:
                        st.info(f"💡 {suggestion}")
                else:
                    st.session_state["df"] = df
                    st.session_state["uploaded_file_name"] = uploaded_file_global.name
                    st.success(f"✅ Loaded: **{uploaded_file_global.name}** ({df.shape[0]} rows × {df.shape[1]} cols)")

            except ValidationError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    with col2:
        meta_file_global = st.file_uploader(
            "🧬 Sample Metadata (Optional)",
            type=["csv", "tsv", "txt", "xlsx", "xls"],
            key="global_metadata_upload",
            help="Upload sample metadata with sample IDs matching your dataset columns"
        )

        if meta_file_global is not None:
            try:
                # Validate file
                FileValidator.validate_upload(meta_file_global)

                # Read file
                meta_df = read_delimited_file(meta_file_global, validate=False)

                # Validate metadata
                is_valid, error, suggestion = DataValidator.validate_metadata(
                    meta_df,
                    expression_df=st.session_state.get("df"),
                    require_sample_col=False
                )

                if not is_valid:
                    st.warning(f"⚠️ {error}")
                    if suggestion:
                        st.info(f"💡 {suggestion}")

                st.session_state["sample_metadata"] = meta_df
                st.session_state["sample_metadata_name"] = meta_file_global.name
                st.success(f"✅ Loaded: **{meta_file_global.name}** ({meta_df.shape[0]} rows × {meta_df.shape[1]} cols)")

            except ValidationError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    with col3:
        st.markdown("**Quick Start**")
        
        # Combined button to load both
        if st.button("🚀 Load Example Dataset + Metadata", use_container_width=True, type="primary", help="Load both example gene expression dataset and metadata for testing"):
            try:
                example_df = pd.read_csv("example_gene_expression.csv")
                example_meta_df = pd.read_csv("example_gene_expression_metadata.csv")
                st.session_state["df"] = example_df
                st.session_state["uploaded_file_name"] = "example_gene_expression.csv"
                st.session_state["sample_metadata"] = example_meta_df
                st.session_state["sample_metadata_name"] = "example_gene_expression_metadata.csv"
                st.success(f"✅ Example dataset and metadata loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Could not load examples: {e}")
        
        # Individual buttons
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            if st.button("📊 Example Dataset", use_container_width=True, help="Load example gene expression dataset only"):
                try:
                    example_df = pd.read_csv("example_gene_expression.csv")
                    st.session_state["df"] = example_df
                    st.session_state["uploaded_file_name"] = "example_gene_expression.csv"
                    st.success(f"✅ Example dataset loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Could not load example: {e}")
        
        with col_ex2:
            if st.button("🧬 Example Metadata", use_container_width=True, help="Load example metadata file only"):
                try:
                    example_meta_df = pd.read_csv("example_gene_expression_metadata.csv")
                    st.session_state["sample_metadata"] = example_meta_df
                    st.session_state["sample_metadata_name"] = "example_gene_expression_metadata.csv"
                    st.success(f"✅ Example metadata loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Could not load example metadata: {e}")

    # Show current data status
    st.markdown("**Current Status:**")
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if st.session_state.get("df") is not None:
            _df = st.session_state["df"]
            st.info(f"✓ Dataset: `{st.session_state.get('uploaded_file_name', '(unnamed)')}` — {_df.shape[0]} rows × {_df.shape[1]} cols")
        else:
            st.warning("⚠️ No dataset loaded")

    with col_status2:
        if st.session_state.get("sample_metadata") is not None:
            meta_df = st.session_state["sample_metadata"]
            st.info(f"✓ Metadata: `{st.session_state.get('sample_metadata_name', '(unnamed)')}` — {meta_df.shape[0]} rows × {meta_df.shape[1]} cols")
        else:
            st.info("ℹ️ No metadata loaded (optional)")

st.markdown("---")

# ========================================================================================
# SIDEBAR (Minimal - Only Gene List Manager)
# ========================================================================================
with st.sidebar:
    st.header("🧬 Tools")
    show_gene_list_manager_sidebar()
    
    # Gene ID Converter
    st.markdown("---")
    with st.expander("🔄 Gene ID Converter", expanded=False):
        from gene_id_converter import show_gene_id_converter
        show_gene_id_converter(expr_df=st.session_state.get("df"))
    
    # Plot Linking
    st.markdown("---")
    with st.expander("🔗 Plot Linking", expanded=False):
        from plot_linking import show_gene_selection_panel, get_linked_selected_genes
        # Get available genes from dataset if available
        available_genes = None
        if st.session_state.get("df") is not None:
            df = st.session_state["df"]
            gene_cols = [c for c in df.columns 
                        if str(c).lower() in ['gene', 'genes', 'symbol', 'gene_symbol', 'id', 'gene_id']]
            if gene_cols:
                available_genes = df[gene_cols[0]].astype(str).tolist()
        show_gene_selection_panel(available_genes=available_genes, key_suffix="_sidebar")
    
    # Project Manager
    st.markdown("---")
    from project_manager import show_project_manager_sidebar
    show_project_manager_sidebar()
    
    # Color theme selector in sidebar
    st.markdown("---")
    from utils.color_themes import get_theme_list, get_recommended_theme
    
    # Initialize color theme if not set
    if "color_theme" not in st.session_state:
        st.session_state["color_theme"] = get_recommended_theme("general")
    
    current_theme = st.session_state.get("color_theme", "Default (tab10)")
    theme_options = get_theme_list()
    theme_index = theme_options.index(current_theme) if current_theme in theme_options else 0
    
    selected_theme = st.selectbox(
        "🎨 Color Theme",
        options=theme_options,
        index=theme_index,
        key="color_theme_selector",
        help="Select color palette for all visualizations"
    )
    
    # Update session state when theme changes
    if selected_theme != current_theme:
        st.session_state["color_theme"] = selected_theme
    
    # Quick theme buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Publication", use_container_width=True, help="Nature-style colors"):
            st.session_state["color_theme"] = get_recommended_theme("publication")
            st.rerun()
    with col2:
        if st.button("👁️ Colorblind", use_container_width=True, help="Colorblind-friendly"):
            st.session_state["color_theme"] = get_recommended_theme("general")
            st.rerun()

# ========================================================================================
# OMICS TYPE SELECTOR (Top Level)
# ========================================================================================
st.markdown("### 🧬 Select Omics Data Type")
omics_type = st.radio(
    "Omics Type:",
    ["RNA-seq (bulk)", "Proteomics", "Genomics (variants)", "Single-cell", "Multi-omics Integration"],
    horizontal=True,
    index=0,
    key="omics_type_selector",
    help="Select the type of omics data you're analyzing"
)

# Store in session state
st.session_state["omics_type"] = omics_type

st.markdown("---")

# ========================================================================================
# HORIZONTAL TABS NAVIGATION - REORGANIZED STRUCTURE
# ========================================================================================
# Use consistent tab structure for RNA-seq and Proteomics (similar workflows)
# Other omics types will be added later
if omics_type in ["RNA-seq (bulk)", "Proteomics"]:
    tab_data_quality, tab_explore, tab_de, tab_functional, tab_advanced, tab_workflows, tab_export, tab_about = st.tabs([
        "📊 Data Quality",
        "🔍 Explore",
        "🔬 Differential Expression",
        "🧬 Functional Analysis",
        "🤖 Advanced Analysis",
        "🧭 Guided Workflows",
        "📦 Export",
        "ℹ️ About"
    ])
elif omics_type == "Genomics (variants)":
    tab_data_quality, tab_explore, tab_analysis, tab_functional, tab_advanced, tab_workflows, tab_export, tab_about = st.tabs([
        "📊 Data Quality",
        "🔍 Explore",
        "🔬 Variant Analysis",
        "🧬 Functional Analysis",
        "🤖 Advanced Analysis",
        "🧭 Guided Workflows",
        "📦 Export",
        "ℹ️ About"
    ])
elif omics_type == "Single-cell":
    tab_data_quality, tab_explore, tab_de, tab_functional, tab_advanced, tab_workflows, tab_export, tab_about = st.tabs([
        "📊 Data Quality",
        "🔍 Explore",
        "🔬 Marker Analysis",
        "🧬 Functional Analysis",
        "🤖 Advanced Analysis",
        "🧭 Guided Workflows",
        "📦 Export",
        "ℹ️ About"
    ])
else:  # Multi-omics Integration
    tab_integration, tab_correlation, tab_visualization, tab_export, tab_about = st.tabs([
        "🔗 Data Integration",
        "📊 Correlation Analysis",
        "📈 Integrated Visualization",
        "📦 Export",
        "ℹ️ About"
    ])

# ========================================================================================
# MAIN TAB 1: DATA QUALITY & PREPARATION
# ========================================================================================
if omics_type in ["RNA-seq (bulk)", "Proteomics"]:
    with tab_data_quality:
        st.markdown("### 📊 Data Quality & Preparation")
        st.caption(f"Quality control, normalization, and data preprocessing for {omics_type} data")
        
        # Subtabs using radio buttons
        subtab_choice = st.radio(
            "Select analysis:",
            ["🧪 QC Dashboard", "🔄 Normalization & Batch Correction"],
            horizontal=True,
            key="data_quality_subtab"
        )
        
        st.markdown("---")
        
        if subtab_choice == "🧪 QC Dashboard":
            if omics_type == "RNA-seq (bulk)":
                if st.session_state["df"] is not None:
                    show_qc_dashboard(st.session_state["df"], metadata_df=st.session_state.get("sample_metadata"))
                else:
                    st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")
            elif omics_type == "Proteomics":
                # Proteomics QC Dashboard
                if st.session_state["df"] is not None:
                    from proteomics_module import show_proteomics_qc_dashboard
                    show_proteomics_qc_dashboard(st.session_state["df"], metadata_df=st.session_state.get("sample_metadata"))
                else:
                    st.warning("⚠️ Please upload a proteomics dataset in the **Shared Data** section above.")
        
        elif subtab_choice == "🔄 Normalization & Batch Correction":
            if st.session_state["df"] is not None:
                if omics_type == "RNA-seq (bulk)":
                    from normalization import show_normalization_tab
                    show_normalization_tab(
                        st.session_state["df"],
                        metadata_df=st.session_state.get("sample_metadata")
                    )
                elif omics_type == "Proteomics":
                    from proteomics_module import show_proteomics_normalization
                    show_proteomics_normalization(
                        st.session_state["df"],
                        metadata_df=st.session_state.get("sample_metadata")
                    )
            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")

# ========================================================================================
# MAIN TAB 2: EXPLORATION & VISUALIZATION
# ========================================================================================
if omics_type in ["RNA-seq (bulk)", "Proteomics"]:
    with tab_explore:
        st.markdown("### 🔍 Exploration & Visualization")
        st.caption("Explore your omics data through dimensionality reduction, gene expression analysis, and comparisons")
        
        # Subtabs using radio buttons
        subtab_choice = st.radio(
            "Select analysis:",
            ["📊 Visualization", "🧬 Gene Expression", "📊 Multi-Condition", "⏱️ Time-Series"],
            horizontal=True,
            key="explore_subtab"
        )
        
        st.markdown("---")
        
        if subtab_choice == "📊 Visualization":
            if st.session_state["df"] is not None:
                df = st.session_state["df"]

                st.markdown(f"**Active dataset:** `{st.session_state['uploaded_file_name']}`")
                st.dataframe(_format_pvals_for_display(df.head(10)), use_container_width=True)
                st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

                # Optional metadata preview
                if st.session_state["sample_metadata"] is not None:
                    with st.expander("📋 Sample Metadata Preview", expanded=False):
                        meta_df = st.session_state["sample_metadata"]
                        st.markdown(f"**Metadata source:** `{st.session_state['sample_metadata_name']}`")
                        st.dataframe(meta_df.head(10), use_container_width=True)
                        st.caption(f"Rows: {len(meta_df)} | Columns: {len(meta_df.columns)}")

                # Show visualizations (works for both RNA-seq and Proteomics)
                if omics_type == "RNA-seq (bulk)":
                    show_visualizations(df)
                elif omics_type == "Proteomics":
                    # Use same visualization module (proteins can be treated similarly to genes)
                    show_visualizations(df)
            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")
        
        elif subtab_choice == "🧬 Gene Expression":
            if st.session_state["df"] is not None:
                if omics_type == "RNA-seq (bulk)":
                    import gene_expression_viewer
                    gene_expression_viewer.show_gene_expression_tab(
                        st.session_state["df"],
                        st.session_state["sample_metadata"]
                    )
                elif omics_type == "Proteomics":
                    # Use same viewer but with protein terminology
                    import gene_expression_viewer
                    st.info("💡 **Protein Expression Viewer** - Analyzing individual protein abundance across samples")
                    gene_expression_viewer.show_gene_expression_tab(
                        st.session_state["df"],
                        st.session_state["sample_metadata"]
                    )
            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")
        
        elif subtab_choice == "📊 Multi-Condition":
            if st.session_state["df"] is not None:
                from multi_condition_comparison import show_multi_condition_comparison_tab
                show_multi_condition_comparison_tab(
                    st.session_state["df"],
                    st.session_state.get("sample_metadata")
                )
            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")
        
        elif subtab_choice == "⏱️ Time-Series":
            if st.session_state["df"] is not None:
                from time_series_module import show_time_series_analysis
                show_time_series_analysis(
                    st.session_state["df"],
                    metadata_df=st.session_state.get("sample_metadata")
                )
            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")

# Single-cell Data Quality tab
if omics_type == "Single-cell":
    with tab_data_quality:
        from single_cell_module import show_single_cell_data_quality_tab
        show_single_cell_data_quality_tab()

# ========================================================================================
# MAIN TAB 3: DIFFERENTIAL EXPRESSION
# ========================================================================================
if omics_type in ["RNA-seq (bulk)", "Proteomics"]:
    with tab_de:
        st.markdown("### 🔬 Differential Expression Analysis")
        if omics_type == "RNA-seq (bulk)":
            st.caption("Identify differentially expressed genes using DESeq2 and visualize results")
        elif omics_type == "Proteomics":
            st.caption("Identify differentially abundant proteins and visualize results")
    
        # Subtabs using radio buttons
        subtab_choice = st.radio(
            "Select analysis:",
            ["🔬 DE Analysis", "📊 DE Plots"],
            horizontal=True,
            key="de_subtab"
        )
        
        st.markdown("---")
        
        if subtab_choice == "🔬 DE Analysis":
            if omics_type == "RNA-seq (bulk)":
                st.info("ℹ️ **Note:** This section requires a **raw count matrix** (integer counts), not the shared normalized dataset.")
                st.caption("For differential expression analysis, upload raw RNA-seq counts (not normalized/log-transformed data)")
                show_counts_de()
            elif omics_type == "Proteomics":
                st.info("ℹ️ **Proteomics DE Analysis** - Identify differentially abundant proteins between conditions")
                st.caption("Upload protein abundance matrix (LFQ, iBAQ, or normalized intensities)")
                from proteomics_module import show_proteomics_de_analysis
                if st.session_state["df"] is not None:
                    show_proteomics_de_analysis(
                        st.session_state["df"],
                        st.session_state.get("sample_metadata")
                    )
                else:
                    st.warning("⚠️ Please upload a proteomics dataset in the **Shared Data** section above.")
        
        elif subtab_choice == "📊 DE Plots":
            if st.session_state["df"] is not None:
                df = st.session_state["df"]
                if omics_type == "RNA-seq (bulk)":
                    from volcano_plot_module import show_de_plots_tab
                    show_de_plots_tab(df)
                elif omics_type == "Proteomics":
                    # Use same volcano plot module (works for proteins too)
                    from volcano_plot_module import show_de_plots_tab
                    st.info("💡 **Protein Volcano Plot** - Visualize differentially abundant proteins")
                    show_de_plots_tab(df)
            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")

# Single-cell Marker Analysis tab
if omics_type == "Single-cell":
    with tab_de:
        from single_cell_module import show_single_cell_marker_analysis_tab
        show_single_cell_marker_analysis_tab()

# ========================================================================================
# MAIN TAB 4: FUNCTIONAL ANALYSIS
# ========================================================================================
if omics_type in ["RNA-seq (bulk)", "Proteomics"]:
    with tab_functional:
        st.markdown("### 🧬 Functional Analysis")
        if omics_type == "RNA-seq (bulk)":
            st.caption("Gene set enrichment, pathway visualization, and gene set overlap analysis")
        elif omics_type == "Proteomics":
            st.caption("Protein set enrichment, pathway visualization, and protein-protein interaction analysis")
        
        # Subtabs using radio buttons - adapt for proteomics
        if omics_type == "Proteomics":
            functional_options = ["🔬 GSEA", "🧬 Pathways", "🔗 Protein Interactions", "🎯 Protein Set Overlaps", "🌊 Flow Diagrams"]
        else:
            functional_options = ["🔬 GSEA", "🧬 Pathways", "🎯 Gene Set Overlaps", "🌊 Flow Diagrams"]
        
        subtab_choice = st.radio(
            "Select analysis:",
            functional_options,
            horizontal=True,
            key="functional_subtab"
        )
        
        st.markdown("---")
        
        if subtab_choice == "🔬 GSEA":
            if omics_type == "RNA-seq (bulk)":
                if st.session_state["df"] is not None:
                    import gsea_module
                    gsea_module.show_gsea_module()
                else:
                    st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")
            elif omics_type == "Proteomics":
                if st.session_state["df"] is not None:
                    import gsea_module
                    st.info("💡 **Protein Set Enrichment Analysis** - Enrichment analysis for protein lists")
                    gsea_module.show_gsea_module()
                else:
                    st.warning("⚠️ Please upload a proteomics dataset in the **Shared Data** section above.")
        
        elif subtab_choice == "🧬 Pathways":
            if st.session_state["df"] is not None:
                from pathway_visualization_module import show_pathway_visualization_tab
                gsea_results = st.session_state.get("gsea_results", None)
                if omics_type == "Proteomics":
                    st.info("💡 **Pathway Visualization with Protein Overlay** - View protein abundance on pathway maps")
                show_pathway_visualization_tab(
                    expr_df=st.session_state["df"],
                    gsea_results=gsea_results
                )
            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")
        
        elif subtab_choice in ["🎯 Gene Set Overlaps", "🎯 Protein Set Overlaps"]:
            if omics_type == "Proteomics":
                st.info("ℹ️ **Note:** This section requires separate DE result files (NOT the shared dataset). Upload 2-5 files to compare protein set overlaps.")
                st.caption("Each file should contain differential abundance results with protein names/IDs")
            else:
                st.info("ℹ️ **Note:** This section requires separate DE result files (NOT the shared dataset). Upload 2-5 files to compare gene set overlaps.")
                st.caption("Each file should contain differential expression results with gene names/IDs")

            # Check if we have example files loaded
            if not st.session_state.get("use_example", False):
                st.warning("⚠️ No DE files loaded yet. Click the button below to load example files or upload your own.")
                st.markdown("### 📂 Quick Start")

                col_venn1, col_venn2 = st.columns(2)
                with col_venn1:
                    if st.button("📂 Auto-Load Example DE Files", use_container_width=True, type="primary"):
                        try:
                            file1 = pd.read_csv("DE_result_1.csv")
                            file2 = pd.read_csv("DE_result_2.csv")
                            file3 = pd.read_csv("DE_result_3.csv")
                            st.session_state["example_DE_files"] = [file1, file2, file3]
                            st.session_state["use_example"] = True
                            st.success("✅ Loaded 3 example DE result files!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Could not load example files: {e}")

                with col_venn2:
                    st.markdown("**Or upload manually** 👇")

            st.markdown("---")

            if st.session_state.get("use_example", False):
                st.success("✅ Using example DE result files (3 files)")
                if st.button("🔄 Switch to manual upload"):
                    st.session_state["use_example"] = False
                    st.session_state.pop("example_DE_files", None)
                    st.rerun()
                if "example_DE_files" in st.session_state:
                    show_overlap_tools(example_dfs=st.session_state["example_DE_files"])
                else:
                    st.warning("⚠️ Example files not loaded properly.")
            else:
                show_overlap_tools()
        
        elif subtab_choice == "🔗 Protein Interactions":
            if omics_type == "Proteomics":
                if st.session_state["df"] is not None:
                    from proteomics_module import show_proteomics_functional_analysis
                    show_proteomics_functional_analysis(
                        st.session_state["df"],
                        st.session_state.get("sample_metadata")
                    )
                else:
                    st.warning("⚠️ Please upload a proteomics dataset in the **Shared Data** section above.")
            else:
                st.info("💡 Protein-protein interaction networks are available for proteomics data. Switch to Proteomics omics type.")
        
        elif subtab_choice == "🌊 Flow Diagrams":
            from alluvial_sankey_module import show_alluvial_sankey_tab
            show_alluvial_sankey_tab(
                st.session_state.get("df"),
                st.session_state.get("sample_metadata")
            )

# ========================================================================================
# MAIN TAB 5: ADVANCED ANALYSIS
# ========================================================================================
if omics_type in ["RNA-seq (bulk)", "Proteomics"]:
    with tab_advanced:
        st.markdown("### 🤖 Advanced Analysis")
        st.caption("Meta-analysis and machine learning for omics data")
        
        # Subtabs using radio buttons
        subtab_choice = st.radio(
            "Select analysis:",
            ["📊 Meta-Analysis", "🤖 ML Classification"],
            horizontal=True,
            key="advanced_subtab"
        )
        
        st.markdown("---")
        
        if subtab_choice == "📊 Meta-Analysis":
            from meta_analysis_module import show_meta_analysis_tab
            show_meta_analysis_tab()
        
        elif subtab_choice == "🤖 ML Classification":
            if st.session_state["df"] is not None:
                from ml_classification_module import show_ml_classification_tab
                if omics_type == "Proteomics":
                    st.info("💡 **Protein-based Classification** - Classify samples using protein abundance signatures")
                show_ml_classification_tab(
                    expr_df=st.session_state["df"],
                    metadata_df=st.session_state.get("sample_metadata")
                )
            else:
                st.warning("⚠️ Please upload a dataset and metadata in the **Shared Data** section above.")

# ========================================================================================
# MAIN TAB 6: EXPORT & PUBLICATION
# ========================================================================================
if omics_type in ["RNA-seq (bulk)", "Proteomics", "Genomics (variants)", "Single-cell"]:
    with tab_export:
        st.markdown("### 📦 Export & Publication")
        st.caption("Create publication-ready figures and export results")
        
        # Subtabs using radio buttons
        subtab_choice = st.radio(
            "Select tool:",
            ["🖼️ Figure Builder", "📦 Export Manager"],
            horizontal=True,
            key="export_subtab"
        )
        
        st.markdown("---")
        
        if subtab_choice == "🖼️ Figure Builder":
            if st.session_state["df"] is not None:
                df = st.session_state["df"]
                st.markdown(f"### 📋 Active Dataset — `{st.session_state['uploaded_file_name']}`")
                st.dataframe(_format_pvals_for_display(df.head(10)), use_container_width=True)
                st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")

                show_multi_panel_figure(df)

                # Optional: show metadata summary alongside figures
                if st.session_state["sample_metadata"] is not None:
                    with st.expander("📋 Sample Metadata (for reference in figure layout)", expanded=False):
                        st.dataframe(st.session_state["sample_metadata"], use_container_width=True)

            else:
                st.warning("⚠️ Please upload a dataset in the **Shared Data** section above.")
        
        elif subtab_choice == "📦 Export Manager":
            show_export_manager()

# ========================================================================================
# MAIN TAB: GUIDED WORKFLOWS
# ========================================================================================
if omics_type in ["RNA-seq (bulk)", "Proteomics", "Genomics (variants)", "Single-cell"]:
    with tab_workflows:
        from guided_workflows import show_workflow_selector
        show_workflow_selector()

# ========================================================================================
# MAIN TAB 7: ABOUT
# ========================================================================================
with tab_about:
    st.title("📘 About OmicsInsight")
    st.markdown("""
    **OmicsInsight** is a modular, interactive bioinformatics visualization suite built for
    **RNA-seq**, **proteomics**, **bulk expression**, and **DE analysis** workflows.
    It empowers life scientists and bioinformaticians to explore data **without writing code**.

    ## 📋 Application Structure

    The application is organized into **7 main sections** following a logical omics analysis workflow:

    ### 📊 **1. Data Quality & Preparation**
    - **QC Dashboard**: Quality control metrics and sample-level checks
    - **Normalization & Batch Correction**: Data normalization and batch effect correction

    ### 🔍 **2. Exploration & Visualization**
    - **Visualization**: PCA, UMAP, heatmaps, correlation networks, advanced clustering
    - **Gene Expression**: Individual gene analysis with statistical comparisons and ANOVA
    - **Multi-Condition**: Pairwise comparisons and comparison heatmaps
    - **Time-Series**: Time-course visualization and trend analysis

    ### 🔬 **3. Differential Expression**
    - **DE Analysis**: DESeq2-based differential expression from raw counts
    - **DE Plots**: Volcano plots, MA plots, waterfall plots with interactive linking

    ### 🧬 **4. Functional Analysis**
    - **GSEA**: Gene set enrichment analysis with Enrichr integration
    - **Pathways**: KEGG/Reactome pathway visualization with expression overlay
    - **Gene Set Overlaps**: Venn diagrams and UpSet plots for overlap analysis
    - **Flow Diagrams**: Alluvial/Sankey diagrams for sample and gene set flows

    ### 🤖 **5. Advanced Analysis**
    - **Meta-Analysis**: Cross-study integration and effect size meta-analysis
    - **ML Classification**: Machine learning for omics-based sample classification

    ### 📦 **6. Export & Publication**
    - **Figure Builder**: Multi-panel figure assembly for publications
    - **Export Manager**: Batch export with publication templates

    ### ℹ️ **7. About**
    - Application information and feature overview

    ---

    ## 🔬 Key Features Overview

    ### 📁 Data Upload & Management
    - Upload **expression matrices**, **metadata**, and multiple **DE result files** (CSV/Excel)
    - Attach sample metadata once and reuse it **across PCA, UMAP, heatmaps, gene viewer, grouped visuals**
    - Auto-detect sample prefixes (e.g., WT_1, KO_2, CTRL-A)

    ---

    ## 🧬 Gene Expression Viewer
    - Search genes by **index**, **gene symbol column**, or **auto-matching**
    - Column filtering options (keep/remove sample columns, remove p-value/logFC columns)
    - Auto-detect sample groups (e.g., WT, KO)
    - Split-view expression (Only WT / Only KO / All Samples)
    - Plot styles: **Boxplot**, **Violin**, **Swarm**
    - Metadata-aware grouping for expression comparisons
    - **Enhanced Statistical Tables**:
      - 95% Confidence Intervals (Welch-Satterthwaite)
      - Log2 Fold Change calculations
      - Effect sizes (Cohen's d)
      - FDR correction (Benjamini-Hochberg)
      - Summary statistics panel
    - **ANOVA for Multi-Group Comparisons** (3+ groups):
      - One-way ANOVA and Kruskal-Wallis tests
      - Post-hoc pairwise comparisons (Bonferroni-corrected)
      - Effect size: Eta-squared (η²)
      - Per-group statistics
    - Export expression table & figure

    ---

    ## 🌋 Volcano Plot Explorer
    - Dynamic thresholds for **p-value** and **log₂FC**
    - Highlight genes via:
      - Manual list
      - Search keyword
      - Uploaded gene list
    - Add plot directly to the **Multi-Panel Figure Builder**
    - PNG/SVG export (publication-ready)
    - **MA Plot** integration (M vs A visualization)
    - **Waterfall Plot** for ranked gene visualization
    - Interactive plot linking across all DE visualizations

    ---

    ## 🎯 Venn / UpSet / Overlap Analysis
    - Compare **2–5 gene sets**
    - Auto-switch between **Venn** (≤3 sets) and **UpSet** (>3 sets)
    - Intersection heatmaps (Overlap count, mean logFC, −log10 p-value)
    - Export:
      - Shared genes
      - Unique genes
      - All pairwise & multi-set combinations

    ---

    ## 🧠 PCA & UMAP Dimensionality Reduction
    - Cluster **genes** or **samples**
    - Color by:
      - Sample group
      - Metadata column
      - KMeans clusters
      - HDBSCAN clusters
    - 2D & 3D interactive scatter plots (with label toggle)
    - **Advanced Clustering Methods**:
      - Hierarchical clustering with dendrograms
      - DBSCAN (density-based clustering)
      - Gaussian Mixture Models (GMM)
      - Silhouette score analysis for optimal cluster number
      - Side-by-side method comparison
    - Download PCA loadings, PCA+UMAP coordinates
    - Save PCA/UMAP plots to Multi-Panel Builder
    - **Custom Color Theme System** (colorblind-friendly palettes)
    - Progress bars for long computations

    ---

    ## 🌡️ Heatmaps, Correlation Maps & Clustermaps
    - Highlight rows (genes/features) using:
      - Manual list
      - Keyword search
      - Uploaded file
    - Transparent dimming + colored outline for visual emphasis
    - **Correlation Network Visualization**:
      - Gene-gene correlation networks (Pearson, Spearman)
      - Community detection (Louvain, Hierarchical)
      - Multiple layout algorithms
      - Node centrality measures
      - Network statistics and export
    - Supports saving all plots to **Figure Builder**
    - PNG/SVG export for publication

    ---

    ## 📊 Grouped Visualizations
    - Bar, Violin, Dot/Swarm, and Line charts
    - Optional hue/group variable
    - Automatic sorting and metadata-aware grouping

    ---

    ## 🖼️ Multi-Panel Figure Builder
    Combine up to **4 plots** into a single publication-ready figure:
    - PCA / UMAP (2D & 3D)
    - Heatmaps / Clustermaps / Correlation
    - Volcano plots
    - Venn / UpSet plots
    - Overlap heatmaps
    Export assembled panels as:
    - **PNG**
    - **PDF**
    - **SVG**

    ---

    ## 🔄 Normalization & Batch Correction
    - Log₂(x+1)
    - Z-score scaling
    - Min-max normalization
    - Column filtering controls
    - **Batch Effect Correction (ComBat-style)**:
      - Empirical Bayes batch correction
      - Parametric and non-parametric options
      - Before/after PCA visualization
      - Automatic batch column detection
      - Optional covariate adjustment

    ---

    ## 💾 Export Options
    - Processed datasets
    - PCA loadings
    - PCA/UMAP coordinates
    - Correlation matrices
    - Overlap gene tables
    - Final composite multi-panel figures

    ---

    ## 📊 Multi-Condition Comparison
    - **Pairwise Comparison Heatmap**: Visualize all pairwise comparisons simultaneously
    - **UpSet Plot for Gene Overlaps**: Identify genes significant in multiple comparisons
    - Summary statistics per comparison
    - Individual comparison results with filtering
    - Flexible group assignment (metadata, auto-detect, manual)
    - Export comparison data and gene membership matrices

    ---

    ## ⏱️ Time-Series Analysis
    - Time-course visualization (line plots with error bars)
    - Automatic time column detection and parsing
    - **Trend Analysis**:
      - Linear regression (parametric)
      - Spearman correlation (non-parametric)
      - Identifies increasing/decreasing/stable trends
    - Multi-gene and multi-condition support
    - Interactive and static plots
    - Export options (PNG, SVG, CSV)

    ---

    ## 🔗 Interactive Plot Linking
    - **Centralized Gene Selection**: Select genes in one plot → automatically highlighted in others
    - Multiple input methods (manual, search, upload, click)
    - Cross-plot highlighting across Volcano, MA, and Waterfall plots
    - Selection mode toggle (add/replace/remove)
    - Download selected genes list
    - Persistent selection across page reruns

    ---

    ## 🌊 Alluvial/Sankey Diagrams
    - **Sample Flow Through Conditions**: Visualize how samples flow through different conditions
    - **Gene Set Transitions**: Visualize overlaps between gene sets
    - **Condition Comparison Flow**: Gene flow between conditions
    - Time-course experiments visualization
    - Pathway relationship mapping
    - Interactive Plotly Sankey diagrams

    ---

    ## 🔄 Gene ID Converter
    - Convert gene IDs between multiple formats:
      - Ensembl ID, Entrez Gene ID, Gene Symbol, RefSeq ID, UniProt ID
    - Multiple input methods (upload, paste, dataset column)
    - Auto-detection of ID type
    - Batch processing (up to 1000 IDs per API call)
    - Species selection (human, mouse, rat, zebrafish, drosophila, c.elegans, yeast)
    - Conversion statistics and results download
    - Uses mygene.info API (free, no API key required)

    ---

    ## 📦 Export Manager
    Streamlined batch export and publication-ready templates:
    - **Batch Export**: Queue multiple figures and tables, download as ZIP archive
    - **Publication Templates**: Pre-configured sizing for major journals
      - Nature/NPG (89mm/183mm)
      - Science (88.9mm/177.8mm)
      - Cell (85mm/174mm)
      - PLOS (83mm/173mm)
      - eLife (88mm/180mm)
    - **Multi-Format Support**:
      - Figures: PNG, PDF, SVG (300 DPI standard, 600 DPI for PLOS)
      - Tables: CSV, Excel (multi-sheet with formatting), LaTeX
    - **Multi-Sheet Excel Builder**: Create formatted workbooks with auto-filters, frozen panes
    - **LaTeX Table Generator**: Publication-ready LaTeX tables with booktabs styling
    - **Golden Ratio Sizing**: Aesthetically pleasing figure dimensions (1.618:1)
    - **Progress Bars**: Visual feedback for long operations (DESeq2, UMAP, etc.)

    ---

    👨‍🔬 **Developed by Dr. Kondaiah Moganti**
    📧 [kmoganti1@gmail.com](mailto:kmoganti1@gmail.com)
    🔗 [linkedin.com/in/dr-kondal-moganti-1748b8149](https://www.linkedin.com/in/dr-kondal-moganti-1748b8149)
    🌐 [omicsinsight.streamlit.app](https://omicsinsight.streamlit.app)
    """)

# ------------------------ Footer ------------------------
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Developed by Dr. Moganti © 2026 | OmicsInsight </div>", unsafe_allow_html=True)
