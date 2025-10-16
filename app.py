"""
Streamlit Single-Cell RNA-seq (scRNA-seq) End-to-End MVP
-------------------------------------------------------
This single-file Streamlit app focuses on the *analysis* pipeline from a counts matrix
(or .h5ad) through QC → normalization → dimensionality reduction → clustering → markers →
(optional) annotation. It optionally exposes a *FASTQ → counts* wrapper via kb-python or
STARsolo if installed on the system, but those steps are best handled offline for scale.

Quick start (local):
1) python -m venv .venv && source .venv/bin/activate
2) pip install -U pip
3) pip install streamlit scanpy anndata pandas numpy matplotlib plotly scikit-learn
   # Optional extras:
   # pip install celltypist scvelo harmonypy bbknn scrublet kb-python
4) streamlit run app.py

Expected inputs:
- .h5ad AnnData (preferred)
- Matrix Market (mtx + barcodes.tsv + genes.tsv/features.tsv.gz)
- 10x HDF5 (.h5) via scanpy.read_10x_h5
- Optional: FASTQ (R1/R2) + reference (kb or STARsolo) — advanced users only

Notes:
- Large datasets: use the “Downsample” option and caching. Running on a beefy machine or a server is recommended.
- This is an MVP and emphasizes clarity over every edge-case.
"""

import os
import io
import json
import tarfile
import zipfile
import tempfile
import subprocess
from pathlib import Path

import streamlit as st

# Lazy imports for heavy deps
import numpy as np
import pandas as pd

# Optional heavy libs are imported when used

st.set_page_config(page_title="scRNA-seq Streamlit MVP", layout="wide")

# ---------------------------
# Utilities
# ---------------------------
# Demo cap settings live in session_state so we can apply right after any load
if "demo_cap_enabled" not in st.session_state:
    st.session_state.demo_cap_enabled = True
if "demo_cap_n" not in st.session_state:
    st.session_state.demo_cap_n = 5000
if "demo_cap_seed" not in st.session_state:
    st.session_state.demo_cap_seed = 0

@st.cache_data(show_spinner=False)
def _cache_bytes(b: bytes):
    """Cache raw bytes and return as-is (helps when reusing uploads)."""
    return b

@st.cache_data(show_spinner=False)
def read_h5ad_from_bytes(raw: bytes):
    import anndata as ad
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        adata = ad.read_h5ad(tmp.name)
    return adata


def _apply_demo_cap(adata, cap: int = 5000, seed: int = 0):
    """Downsample cells to `cap` if enabled and adata.n_obs > cap.
    Returns (adata2, applied_bool).
    """
    if not st.session_state.get("demo_cap_enabled", True):
        return adata, False
    if adata.n_obs <= int(cap):
        return adata, False
    rs = np.random.RandomState(int(seed))
    idx = rs.choice(adata.n_obs, int(cap), replace=False)
    adata2 = adata[idx, :].copy()
    adata2.uns["_demo_cap"] = {
        "cap": int(cap),
        "seed": int(seed),
        "from_n_obs": int(adata.n_obs),
    }
    return adata2, True
@st.cache_data(show_spinner=False)
def _cache_bytes(b: bytes):
    """Cache raw bytes and return as-is (helps when reusing uploads)."""
    return b

@st.cache_data(show_spinner=False)
def read_h5ad_from_bytes(raw: bytes):
    import anndata as ad
    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        adata = ad.read_h5ad(tmp.name)
    return adata

@st.cache_data(show_spinner=False)
def read_10x_mtx(archive_bytes: bytes, inner_dir_hint: str | None = None):
    import scanpy as sc
    # Accept a .zip or .tar.gz upload containing matrix.mtx, barcodes, features/genes
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        buf = io.BytesIO(archive_bytes)
        # Try tar first, then zip
        try:
            with tarfile.open(fileobj=buf, mode="r:gz") as tf:
                tf.extractall(tmpdir)
        except tarfile.ReadError:
            buf.seek(0)
            with zipfile.ZipFile(buf) as zf:
                zf.extractall(tmpdir)
        # Heuristics to find the matrix folder
        candidate = None
        for p in tmpdir.rglob("matrix.mtx*"):
            candidate = p.parent
            break
        if inner_dir_hint and (tmpdir / inner_dir_hint).exists():
            candidate = tmpdir / inner_dir_hint
        if candidate is None:
            raise FileNotFoundError("Could not locate matrix.mtx in the uploaded archive.")
        adata = sc.read_10x_mtx(candidate, var_names='gene_symbols', cache=False)
    return adata

@st.cache_data(show_spinner=False)
def read_10x_h5(raw: bytes):
    import scanpy as sc
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        adata = sc.read_10x_h5(tmp.name)
    return adata


def ensure_plotly_registered():
    import plotly.io as pio
    pio.templates.default = "plotly"


def _umap_scatter(adata, key: str = "leiden"):
    import plotly.express as px
    ensure_plotly_registered()
    if "X_umap" not in adata.obsm:
        st.warning("UMAP not computed yet. Run Dimensionality Reduction.")
        return None
    df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"]).copy()
    if key in adata.obs.columns:
        df[key] = adata.obs[key].astype(str).values
    else:
        df[key] = "NA"
    fig = px.scatter(df, x="UMAP1", y="UMAP2", color=key, hover_data=[df.index])
    return fig


def _rank_genes_df(adata, groupby="leiden"):
    # Convert scanpy rank_genes_groups result into a tidy DataFrame
    rg = adata.uns.get("rank_genes_groups")
    if rg is None:
        return pd.DataFrame()
    groups = rg["names"].dtype.names
    out = []
    for g in groups:
        names = pd.Series(rg["names"][g]).astype(str)
        scores = pd.Series(rg.get("scores", {}).get(g, [np.nan] * len(names))) if isinstance(rg.get("scores"), dict) else pd.Series(np.array(rg.get("scores"))[:, list(groups).index(g)]) if rg.get("scores") is not None else pd.Series([np.nan]*len(names))
        pvals = pd.Series(rg.get("pvals_adj", {}).get(g, [np.nan] * len(names))) if isinstance(rg.get("pvals_adj"), dict) else pd.Series(np.array(rg.get("pvals_adj"))[:, list(groups).index(g)]) if rg.get("pvals_adj") is not None else pd.Series([np.nan]*len(names))
        for n, s, p in zip(names, scores, pvals):
            out.append({"group": g, "gene": n, "score": s, "pval_adj": p})
    return pd.DataFrame(out)


# ---------------------------
# Sidebar: Data Ingest
# ---------------------------
st.sidebar.title("Pipeline")
# Demo limits (affects any newly loaded data)
st.sidebar.markdown("### Demo limits")
st.session_state.demo_cap_enabled = st.sidebar.checkbox("Enable demo cell cap", value=st.session_state.demo_cap_enabled)
st.session_state.demo_cap_n = int(st.sidebar.number_input("Max cells (cap)", min_value=500, max_value=50000, value=st.session_state.demo_cap_n, step=500))
st.session_state.demo_cap_seed = int(st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=st.session_state.demo_cap_seed, step=1))
step = st.sidebar.radio(
    "Choose step",
    [
        "Load Data",
        "QC & Filtering",
        "Normalize & HVGs",
        "Dimensionality Reduction",
        "Clustering",
        "Markers & DE",
        "Cell Type Annotation (optional)",
        "(Optional) Trajectory",
        "Export",
    ],
)

st.title("Single-Cell RNA-seq – Streamlit MVP")
st.caption("Upload data, run QC → clustering, and explore results interactively. ✨")

if "adata" not in st.session_state:
    st.session_state.adata = None

# ---------------------------
# Step 1: Load Data
# ---------------------------
if step == "Load Data":
    st.subheader("Upload or Load Data")

    # Demo loader (if local data folder present)
    import scanpy as sc
    from pathlib import Path
    st.markdown("### 📊 Load Demo Datasets")
    demo_files = {
        "Synthetic Demo (2000 cells × 1000 genes)": "data/demo_2000cells_1000genes.h5ad",
        "PBMC 1k v3 (10x HDF5)": "data/pbmc_1k_v3_filtered_feature_bc_matrix.h5",
    }
    demo_choice = st.selectbox("Choose a demo dataset:", list(demo_files.keys()))
    if st.button("Load Selected Demo"):
        path = Path(demo_files[demo_choice])
        if not path.exists():
            st.error(f"Demo file not found: {path}")
        else:
            try:
                if path.suffix == ".h5ad":
                    adata = sc.read_h5ad(path)
                elif path.suffix == ".h5":
                    adata = sc.read_10x_h5(path)
                else:
                    st.error("Unsupported file format.")
                    st.stop()
                # Make gene names unique
                try:
                    adata.var_names_make_unique()
                except Exception:
                    pass
                # Apply demo cap
                adata, applied = _apply_demo_cap(adata, cap=st.session_state.demo_cap_n, seed=st.session_state.demo_cap_seed)
                st.session_state.adata = adata
                st.success(f"✅ Loaded {demo_choice}: {adata.n_obs} cells × {adata.n_vars} genes.")
                if applied:
                    st.info(f"Demo cap applied: downsampled from {adata.uns['_demo_cap']['from_n_obs']} → {adata.n_obs} cells.")
            except Exception as e:
                st.error(f"Failed to load demo: {e}")

    tabs = st.tabs([".h5ad", "10x MTX (.zip/.tar.gz)", "10x HDF5 (.h5)", "FASTQ → counts (advanced)"])

    with tabs[0]:
        h5ad_file = st.file_uploader("Upload .h5ad", type=["h5ad"])  
        if h5ad_file is not None:
            raw = _cache_bytes(h5ad_file.read())
            with st.spinner("Reading .h5ad..."):
                adata = read_h5ad_from_bytes(raw)
            st.success(f"Loaded AnnData with {adata.n_obs} cells × {adata.n_vars} genes.")
            # Ensure unique gene names to avoid warnings
            try:
                adata.var_names_make_unique()
            except Exception:
                pass
            # Apply demo cap immediately after load
            adata, applied = _apply_demo_cap(adata, cap=st.session_state.demo_cap_n, seed=st.session_state.demo_cap_seed)
            st.session_state.adata = adata
            if applied:
                st.info(f"Demo cap applied: downsampled to {adata.n_obs} cells (from {adata.uns['_demo_cap']['from_n_obs']}).")

    with tabs[1]:
        mtx_archive = st.file_uploader("Upload 10x matrix archive (.zip or .tar.gz)", type=["zip", "gz", "tar", "tgz"])  
        hint = st.text_input("Inner folder hint (optional)", value="")
        if mtx_archive is not None:
            raw = _cache_bytes(mtx_archive.read())
            with st.spinner("Reading 10x MTX..."):
                adata = read_10x_mtx(raw, inner_dir_hint=hint or None)
            st.success(f"Loaded matrix with {adata.n_obs} cells × {adata.n_vars} genes.")
            try:
                adata.var_names_make_unique()
            except Exception:
                pass
            # Apply demo cap immediately after load
            adata, applied = _apply_demo_cap(adata, cap=st.session_state.demo_cap_n, seed=st.session_state.demo_cap_seed)
            st.session_state.adata = adata
            if applied:
                st.info(f"Demo cap applied: downsampled to {adata.n_obs} cells (from {adata.uns['_demo_cap']['from_n_obs']}).")

    with tabs[2]:
        h5_file = st.file_uploader("Upload 10x HDF5 (.h5)", type=["h5"])  
        if h5_file is not None:
            raw = _cache_bytes(h5_file.read())
            with st.spinner("Reading 10x HDF5..."):
                adata = read_10x_h5(raw)
            st.success(f"Loaded matrix with {adata.n_obs} cells × {adata.n_vars} genes.")
            try:
                adata.var_names_make_unique()
            except Exception:
                pass
            # Apply demo cap immediately after load
            adata, applied = _apply_demo_cap(adata, cap=st.session_state.demo_cap_n, seed=st.session_state.demo_cap_seed)
            st.session_state.adata = adata
            if applied:
                st.info(f"Demo cap applied: downsampled to {adata.n_obs} cells (from {adata.uns['_demo_cap']['from_n_obs']}).")

    with tabs[3]:
        st.warning("Advanced: requires kb-python or STARsolo pre-installed on the host. This may be slow/expensive.")
        st.warning("Advanced: requires kb-python or STARsolo pre-installed on the host. This may be slow/expensive.")
        run_mode = st.selectbox("Backend", ["kb-python (kallisto|bustools)", "STARsolo"], index=0)
        fq1 = st.file_uploader("Upload R1 FASTQ (gz)", type=["fastq.gz"], key="fq1")
        fq2 = st.file_uploader("Upload R2 FASTQ (gz)", type=["fastq.gz"], key="fq2")
        reference_path = st.text_input("Reference index path (prebuilt)")
        out_name = st.text_input("Output prefix", value="kb_out")
        run = st.button("Run FASTQ → counts")
        if run and fq1 and fq2 and reference_path:
            fq1_b = _cache_bytes(fq1.read())
            fq2_b = _cache_bytes(fq2.read())
            with tempfile.TemporaryDirectory() as td:
                r1p = Path(td)/"R1.fastq.gz"; r1p.write_bytes(fq1_b)
                r2p = Path(td)/"R2.fastq.gz"; r2p.write_bytes(fq2_b)
                outdir = Path(td)/"out"; outdir.mkdir(exist_ok=True)
                if run_mode.startswith("kb"):
                    cmd = [
                        "kb", "count", "-i", str(Path(reference_path)/"index.idx"),
                        "-g", str(Path(reference_path)/"transcripts_to_genes.txt"),
                        "-x", "10xv3", "-o", str(outdir), str(r1p), str(r2p)
                    ]
                else:
                    cmd = [
                        "STAR", "--runThreadN", "8",
                        "--soloType", "CB_UMI_Simple",
                        "--soloCBwhitelist", "None",
                        "--genomeDir", reference_path,
                        "--readFilesIn", str(r1p), str(r2p),
                        "--readFilesCommand", "zcat",
                        "--soloFeatures", "Gene",
                        "--outFileNamePrefix", str(Path(td)/out_name)
                    ]
                st.info("Running: " + " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                    # Try to read 10x output if present
                    import scanpy as sc
                    mtx_dir = None
                    for p in outdir.rglob("matrix.mtx*"):
                        mtx_dir = p.parent
                        break
                    if mtx_dir is None:
                        raise RuntimeError("Did not find matrix.mtx in output")
                    adata = sc.read_10x_mtx(mtx_dir, var_names='gene_symbols', cache=False)
                    st.session_state.adata = adata
                    st.success(f"Loaded counts with {adata.n_obs} cells × {adata.n_vars} genes.")
                    # Apply demo cap immediately after load
                    adata, applied = _apply_demo_cap(adata, cap=st.session_state.demo_cap_n, seed=st.session_state.demo_cap_seed)
                    st.session_state.adata = adata
                    if applied:
                        st.info(f"Demo cap applied: downsampled to {adata.n_obs} cells (from {adata.uns['_demo_cap']['from_n_obs']}).")
                except Exception as e:
                    st.error(f"FASTQ → counts failed: {e}")

    if st.session_state.adata is not None:
        st.dataframe(st.session_state.adata.obs.head(), use_container_width=True)

# ---------------------------
# Step 2: QC & Filtering
# ---------------------------
if step == "QC & Filtering":
    if st.session_state.adata is None:
        st.stop()

    import scanpy as sc
    import numpy as np

    adata = st.session_state.adata.copy()
    st.subheader("Quality Control")

    gene_prefix = st.text_input(
        "Mitochondrial gene prefix (human: 'MT-', mouse: 'mt-')",
        value="MT-",
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

    # Compute QC metrics
    if st.button("Compute QC metrics"):
        try:
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
            st.success("✅ QC metrics computed successfully.")
        except KeyError:
            st.error("Missing 'mt' annotation. Check mitochondrial gene prefix.")
        except Exception as e:
            st.error(f"QC failed: {e}")

    # Filtering parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        n_genes_min = st.number_input("Min genes per cell", value=200, step=50)
    with col2:
        n_genes_max = st.number_input("Max genes per cell", value=6000, step=500)
    with col3:
        mt_max = st.slider("Max mito %", min_value=0, max_value=100, value=20)

    if st.button("Filter cells/genes"):
        before = (adata.n_obs, adata.n_vars)
        if "n_genes_by_counts" not in adata.obs or "pct_counts_mt" not in adata.obs:
            try:
                sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
            except Exception as e:
                st.error(f"QC metrics missing and failed to compute: {e}")
                st.stop()

        sc.pp.filter_cells(adata, min_genes=int(n_genes_min))
        adata = adata[adata.obs["n_genes_by_counts"] <= int(n_genes_max)].copy()
        if "pct_counts_mt" in adata.obs:
            adata = adata[adata.obs["pct_counts_mt"] <= float(mt_max)].copy()
        sc.pp.filter_genes(adata, min_cells=3)

        after = (adata.n_obs, adata.n_vars)
        st.info(f"Filtered: {before} → {after}")
        st.session_state.adata = adata

    # Show QC summary table
    qc_cols = [
        c
        for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
        if c in adata.obs.columns
    ]
    if qc_cols:
        st.write("Top cells by QC metrics:")
        st.dataframe(adata.obs[qc_cols].head(), width="stretch")


# ---------------------------

# Step 3: Normalize & HVGs
# ---------------------------
if step == "Normalize & HVGs":
    if st.session_state.adata is None:
        st.stop()
    import scanpy as sc
    adata = st.session_state.adata.copy()

    st.subheader("Normalization")
    method = st.selectbox("Method", ["pp.normalize_total + log1p", "SCTransform (optional)"])
    if method.startswith("pp.normalize_total"):
        target_sum = st.number_input("Target sum per cell", value=1e4, step=1e3, format="%.0f")
        if st.button("Run normalization"):
            sc.pp.normalize_total(adata, target_sum=target_sum)
            sc.pp.log1p(adata)
            st.success("Normalized and log1p-transformed.")
    else:
        st.info("SCTransform not included by default. Consider sctransform via scvi-tools or Seurat interop.")

    st.subheader("Highly Variable Genes")
    # Use safer default flavor for Streamlit Cloud to avoid numba ImportError on Python 3.13
    flavor_choice = st.selectbox(
        "HVG flavor",
        ["cell_ranger", "seurat", "seurat_v3 (needs numba)"]
    )
    flavor_map = {
        "cell_ranger": "cell_ranger",
        "seurat": "seurat",
        "seurat_v3 (needs numba)": "seurat_v3",
    }
    flavor = flavor_map[flavor_choice]
    n_top = st.number_input("n_top_genes", value=2000, step=500)
    if st.button("Find HVGs"):
        try:
            sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=int(n_top))
            st.write(adata.var.get("highly_variable", pd.Series(index=adata.var_names)).value_counts())
        except ImportError as e:
          st.warning(f"{e}. Falling back to flavor='seurat' (no numba needed).")
          sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(n_top))

Falling back to flavor='seurat' (no numba needed).")
            sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=int(n_top))
            st.write(adata.var.get("highly_variable", pd.Series(index=adata.var_names)).value_counts())
        except Exception as e:
            st.error(f"HVG computation failed: {e}")

    if st.button("Save and continue"):
        st.session_state.adata = adata
        st.success("Saved updates to session.")

# ---------------------------
# Step 4: Dimensionality Reduction
# ---------------------------
if step == "Dimensionality Reduction":
    if st.session_state.adata is None:
        st.stop()
    import scanpy as sc
    adata = st.session_state.adata.copy()

    st.subheader("PCA / Neighbors / UMAP")
    use_hvg = st.checkbox("Use only HVGs", value=True)
    n_pcs = st.slider("Number of PCs", 10, 100, 50)
    neighbors_k = st.slider("Neighbors k", 5, 50, 15)
    if st.button("Run DR"):
        if use_hvg and "highly_variable" in adata.var.columns:
            adata = adata[:, adata.var["highly_variable"]].copy()
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, n_comps=int(n_pcs))
        sc.pp.neighbors(adata, n_neighbors=int(neighbors_k), n_pcs=int(n_pcs))
        sc.tl.umap(adata)
        st.success("Computed PCA, neighbors, and UMAP.")
    if st.button("Save and continue"):
        st.session_state.adata = adata
    fig = _umap_scatter(adata)
    if fig is not None:
        st.plotly_chart(fig, width='stretch')

# ---------------------------
# Step 5: Clustering
# ---------------------------
if step == "Clustering":
    if st.session_state.adata is None:
        st.stop()
    import scanpy as sc
    from sklearn.cluster import KMeans
    adata = st.session_state.adata.copy()

    st.subheader("Clustering")
    algo = st.selectbox(
        "Algorithm",
        [
            "leiden (needs igraph)",
            "louvain (needs igraph)",
            "KMeans (no extra deps)",
        ],
        index=0,
    )

    # Common controls
    resolution = st.slider("Resolution (graph methods)", 0.1, 2.0, 0.5, 0.1)
    k_kmeans = st.slider("K (for KMeans)", 2, 50, 10)

    if st.button("Run clustering"):
        try:
            if algo.startswith("leiden"):
                sc.tl.leiden(adata, resolution=float(resolution))
                st.success("Leiden clustering done.")
            elif algo.startswith("louvain"):
                sc.tl.louvain(adata, resolution=float(resolution))
                st.success("Louvain clustering done.")
            else:
                # KMeans on PCA (compute minimal PCA if needed)
                if "X_pca" not in adata.obsm:
                    st.info("PCA not found; computing PCA (50 comps) quickly for KMeans…")
                    sc.pp.scale(adata, max_value=10)
                    sc.tl.pca(adata, n_comps=50)
                X = adata.obsm["X_pca"]
                labels = KMeans(n_clusters=int(k_kmeans), n_init=10, random_state=0).fit_predict(X)
                adata.obs["kmeans"] = pd.Categorical(labels.astype(str))
                st.success("KMeans clustering done.")
        except ImportError as e:
            # Fall back to KMeans if igraph/leidenalg not available
            st.warning(f"{e}. Falling back to KMeans (no igraph needed).")
            if "X_pca" not in adata.obsm:
                sc.pp.scale(adata, max_value=10)
                sc.tl.pca(adata, n_comps=50)
            X = adata.obsm["X_pca"]
            labels = KMeans(n_clusters=int(k_kmeans), n_init=10, random_state=0).fit_predict(X)
            adata.obs["kmeans"] = pd.Categorical(labels.astype(str))
            st.success("KMeans clustering done.")
        except Exception as e:
            st.error(f"Clustering failed: {e}")
        else:
            st.session_state.adata = adata

    # Determine key for plotting counts
    key = None
    for c in ["leiden", "louvain", "kmeans"]:
        if c in adata.obs.columns:
            key = c
            break
    if key:
        fig = _umap_scatter(adata, key)
        if fig is not None:
            st.plotly_chart(fig, width='stretch')
        st.dataframe(adata.obs[key].value_counts().rename_axis("cluster").reset_index(name="n"))

# ---------------------------

# Step 6: Markers & DE
# ---------------------------
if step == "Markers & DE":
    if st.session_state.adata is None:
        st.stop()
    import scanpy as sc
    adata = st.session_state.adata.copy()

    st.subheader("Rank genes per cluster")
    groupby = st.selectbox("Group by", [c for c in adata.obs.columns if adata.obs[c].dtype.name in ["category","object","int","int64"]], index=max(0, [*adata.obs.columns].index("leiden") if "leiden" in adata.obs.columns else 0))
    method = st.selectbox("Method", ["wilcoxon", "t-test", "logreg"], index=0)
    top_n = st.slider("Top N per group", 5, 100, 25)
    if st.button("Run DE"):
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)
        df = _rank_genes_df(adata, groupby)
        if not df.empty:
            # Show top_n per group
            df_top = df.sort_values(["group", "pval_adj", "score"], ascending=[True, True, False]).groupby("group").head(int(top_n))
            st.dataframe(df_top, width='stretch')
        else:
            st.info("No DE results found.")
    if st.button("Save and continue"):
        st.session_state.adata = adata

# ---------------------------
# Step 7: Cell Type Annotation
# ---------------------------
if step == "Cell Type Annotation (optional)":
    if st.session_state.adata is None:
        st.stop()
    adata = st.session_state.adata.copy()
    st.subheader("CellTypist (optional)")
    st.caption("Requires `pip install celltypist` and a downloaded model.")
    model_path = st.text_input("CellTypist model path (e.g., Immune_All_Low.pkl)", value="")
    if st.button("Run CellTypist"):
        try:
            import celltypist
            from celltypist import models
            mdl = model_path if model_path else models.download_models(model="Immune_All_Low")
            res = celltypist.annotate(adata, model=mdl)
            adata.obs["celltypist_labels"] = res.predicted_labels
            st.success("Annotation complete. Added `celltypist_labels` to adata.obs.")
        except Exception as e:
            st.error(f"CellTypist failed: {e}")
    if st.button("Save and continue"):
        st.session_state.adata = adata
    # Show UMAP colored by labels if available
    key = "celltypist_labels" if "celltypist_labels" in adata.obs else None
    if key:
        fig = _umap_scatter(adata, key)
        if fig is not None:
            st.plotly_chart(fig, width='stretch')

# ---------------------------
# Step 8: (Optional) Trajectory
# ---------------------------
if step == "(Optional) Trajectory":
    if st.session_state.adata is None:
        st.stop()
    adata = st.session_state.adata.copy()
    st.subheader("scVelo (RNA velocity)")
    st.caption("Requires spliced/unspliced layers and `pip install scvelo`. This is an optional preview.")
    mode = st.selectbox("Mode", ["stochastic", "dynamical"], index=1)
    if st.button("Run velocity"):
        try:
            import scvelo as scv
            scv.pp.filter_and_normalize(adata)
            scv.pp.moments(adata)
            if mode == "stochastic":
                scv.tl.velocity(adata)
                scv.tl.velocity_graph(adata)
            else:
                scv.tl.recover_dynamics(adata)
                scv.tl.velocity(adata, mode="dynamical")
                scv.tl.velocity_graph(adata)
            st.success("Velocity computed.")
        except Exception as e:
            st.error(f"Velocity failed: {e}")
    if st.button("Save and continue"):
        st.session_state.adata = adata

# ---------------------------
# Step 9: Export
# ---------------------------
if step == "Export":
    if st.session_state.adata is None:
        st.stop()
    import anndata as ad
    adata = st.session_state.adata
    st.subheader("Export results")
    fname = st.text_input("File name", value="analysis.h5ad")
    if st.button("Package .h5ad for download"):
        with tempfile.TemporaryDirectory() as td:
            outp = Path(td)/fname
            adata.write(outp)
            with open(outp, "rb") as f:
                st.download_button("Download .h5ad", data=f, file_name=fname, mime="application/octet-stream")
    st.caption("Tip: also export cluster markers as CSV from the DE tab.")

# Footer
st.markdown("""
---
**Disclaimers**
- This MVP is for research exploration only.
- Alignment/counting from raw FASTQs is resource-intensive; consider using Cell Ranger, STARsolo, or kb-python offline and loading counts here.
- Always validate results with domain knowledge and replicate analyses.
""")
