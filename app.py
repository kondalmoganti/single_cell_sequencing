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
import plotly.express as px
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


def _umap_scatter(adata, color_key=None):
    if "X_umap" not in adata.obsm:
        return None
    emb = adata.obsm["X_umap"]
    df = pd.DataFrame({"UMAP1": emb[:, 0], "UMAP2": emb[:, 1]})
    if color_key is not None and color_key in adata.obs:
        df["color"] = adata.obs[color_key].astype(str)
        color_arg = "color"
    else:
        color_arg = None

    fig = px.scatter(
        df,
        x="UMAP1", y="UMAP2",
        color=color_arg,
        hover_data=None,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
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
        "Embedding & Clustering",
        "Markers & DE",
        "Cell Type Annotation",
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
    import numpy as np

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
                    # add minimal QC columns so preview isn't empty
                    X = adata.X
                    try:
                        total = np.asarray(X.sum(axis=1)).ravel()
                        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
                    except Exception:
                        total = X.sum(axis=1)
                        n_genes = (X > 0).sum(axis=1)
                    adata.obs["total_counts"] = total
                    adata.obs["n_genes_by_counts"] = n_genes
                else:
                    st.error("Unsupported file format.")
                    st.stop()

                # Make gene names unique
                try:
                    adata.var_names_make_unique()
                except Exception:
                    pass

                # Apply demo cap
                adata, applied = _apply_demo_cap(
                    adata,
                    cap=st.session_state.demo_cap_n,
                    seed=st.session_state.demo_cap_seed,
                )
                st.session_state.adata = adata
                st.success(f"✅ Loaded {demo_choice}: {adata.n_obs} cells × {adata.n_vars} genes.")
                if applied:
                    st.info(
                        f"Demo cap applied: downsampled from "
                        f"{adata.uns['_demo_cap']['from_n_obs']} → {adata.n_obs} cells."
                    )
            except Exception as e:
                st.error(f"Failed to load demo: {e}")

    tabs = st.tabs([".h5ad", "10x MTX (.zip/.tar.gz)", "10x HDF5 (.h5)", "FASTQ → counts (advanced)"])

    # -------- .h5ad upload
    with tabs[0]:
        h5ad_file = st.file_uploader("Upload .h5ad", type=["h5ad"])
        if h5ad_file is not None:
            raw = _cache_bytes(h5ad_file.read())
            with st.spinner("Reading .h5ad..."):
                adata = read_h5ad_from_bytes(raw)
            st.success(f"Loaded AnnData with {adata.n_obs} cells × {adata.n_vars} genes.")
            try:
                adata.var_names_make_unique()
            except Exception:
                pass
            adata, applied = _apply_demo_cap(
                adata,
                cap=st.session_state.demo_cap_n,
                seed=st.session_state.demo_cap_seed,
            )
            st.session_state.adata = adata
            if applied:
                st.info(
                    f"Demo cap applied: downsampled to {adata.n_obs} cells "
                    f"(from {adata.uns['_demo_cap']['from_n_obs']})."
                )

    # -------- 10x MTX archive upload
    with tabs[1]:
        mtx_archive = st.file_uploader(
            "Upload 10x matrix archive (.zip or .tar.gz)",
            type=["zip", "gz", "tar", "tgz"],
        )
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
            adata, applied = _apply_demo_cap(
                adata,
                cap=st.session_state.demo_cap_n,
                seed=st.session_state.demo_cap_seed,
            )
            st.session_state.adata = adata
            if applied:
                st.info(
                    f"Demo cap applied: downsampled to {adata.n_obs} cells "
                    f"(from {adata.uns['_demo_cap']['from_n_obs']})."
                )

    # -------- 10x HDF5 upload
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
            # add minimal QC columns so preview isn't empty
            X = adata.X
            try:
                total = np.asarray(X.sum(axis=1)).ravel()
                n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
            except Exception:
                total = X.sum(axis=1)
                n_genes = (X > 0).sum(axis=1)
            adata.obs["total_counts"] = total
            adata.obs["n_genes_by_counts"] = n_genes

            adata, applied = _apply_demo_cap(
                adata,
                cap=st.session_state.demo_cap_n,
                seed=st.session_state.demo_cap_seed,
            )
            st.session_state.adata = adata
            if applied:
                st.info(
                    f"Demo cap applied: downsampled to {adata.n_obs} cells "
                    f"(from {adata.uns['_demo_cap']['from_n_obs']})."
                )

    # -------- FASTQ → counts (advanced)
    with tabs[3]:
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
                r1p = Path(td) / "R1.fastq.gz"; r1p.write_bytes(fq1_b)
                r2p = Path(td) / "R2.fastq.gz"; r2p.write_bytes(fq2_b)
                outdir = Path(td) / "out"; outdir.mkdir(exist_ok=True)
                if run_mode.startswith("kb"):
                    cmd = [
                        "kb", "count",
                        "-i", str(Path(reference_path) / "index.idx"),
                        "-g", str(Path(reference_path) / "transcripts_to_genes.txt"),
                        "-x", "10xv3",
                        "-o", str(outdir),
                        str(r1p), str(r2p),
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
                        "--outFileNamePrefix", str(Path(td) / out_name),
                    ]
                st.info("Running: " + " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                    # Try to read 10x output if present
                    mtx_dir = None
                    for p in outdir.rglob("matrix.mtx*"):
                        mtx_dir = p.parent
                        break
                    if mtx_dir is None:
                        raise RuntimeError("Did not find matrix.mtx in output")
                    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
                    # preview columns
                    X = adata.X
                    try:
                        total = np.asarray(X.sum(axis=1)).ravel()
                        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
                    except Exception:
                        total = X.sum(axis=1)
                        n_genes = (X > 0).sum(axis=1)
                    adata.obs["total_counts"] = total
                    adata.obs["n_genes_by_counts"] = n_genes
                    st.success(f"Loaded counts with {adata.n_obs} cells × {adata.n_vars} genes.")

                    # Apply demo cap
                    adata, applied = _apply_demo_cap(
                        adata,
                        cap=st.session_state.demo_cap_n,
                        seed=st.session_state.demo_cap_seed,
                    )
                    st.session_state.adata = adata
                    if applied:
                        st.info(
                            f"Demo cap applied: downsampled to {adata.n_obs} cells "
                            f"(from {adata.uns['_demo_cap']['from_n_obs']})."
                        )
                except Exception as e:
                    st.error(f"FASTQ → counts failed: {e}")

    # -------- Preview (always show something)
    if st.session_state.adata is not None:
        adata = st.session_state.adata
        # ensure unique gene names
        try:
            adata.var_names_make_unique()
        except Exception:
            pass

        st.markdown("### Preview")
        st.write(f"**Shape:** {adata.n_obs} cells × {adata.n_vars} genes")

        # show per-cell table or barcodes if empty
        if adata.obs.shape[1] == 0:
            st.write("No per-cell metadata yet (10x raw). Run **QC & Filtering** to compute QC columns.")
            preview_obs = adata.obs_names.to_series().head().to_frame(name="barcode")
            st.dataframe(preview_obs, width="stretch")
        else:
            st.dataframe(adata.obs.head(), width="stretch")

        st.write("Gene table (first 5):")
        if adata.var.shape[1] == 0:
            st.dataframe(adata.var_names.to_series().head().to_frame(name="gene"), width="stretch")
        else:
            st.dataframe(adata.var.head(), width="stretch")

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
    import pandas as pd
    import numpy as np

    adata = st.session_state.adata.copy()

    # ---------- Normalization ----------
    st.subheader("Normalization")

    method = st.selectbox(
        "Method",
        ["pp.normalize_total + log1p", "SCVI (variance-stabilizing)"],
        key="norm_method",
    )

    if method == "pp.normalize_total + log1p":
        target_sum = st.number_input(
            "Target sum per cell",
            value=1e4, step=1e3, format="%.0f", key="norm_target_sum"
        )
        if st.button("Run normalization", key="norm_run"):
            sc.pp.normalize_total(adata, target_sum=float(target_sum))
            sc.pp.log1p(adata)
            st.success("✅ Normalized and log1p-transformed.")
    else:
        # SCVI path (Python 3.11 recommended; requires scvi-tools/torch)
        try:
            import scvi

            # Ensure counts layer exists
            if "counts" not in adata.layers:
                if adata.raw is not None and adata.raw.X is not None:
                    adata.layers["counts"] = adata.raw.X.copy()
                else:
                    adata.layers["counts"] = adata.X.copy()

            scvi.settings.seed = 0
            scvi.model.SCVI.setup_anndata(adata, layer="counts")

            n_latent = st.number_input(
                "Latent dim (SCVI)", min_value=8, max_value=64, value=30, step=2,
                key="scvi_n_latent"
            )
            max_epochs = st.number_input(
                "Max epochs", min_value=50, max_value=500, value=200, step=50,
                key="scvi_epochs"
            )

            if st.button("Train SCVI", key="scvi_train"):
                with st.spinner("Training SCVI (CPU)…"):
                    model = scvi.model.SCVI(adata, n_latent=int(n_latent))
                    model.train(max_epochs=int(max_epochs), early_stopping=True,
                                plan_kwargs={"weight_decay": 0.0})
                # Store outputs
                adata.obsm["X_scvi"] = model.get_latent_representation()

                norm = model.get_normalized_expression(library_size=1e4)
                try:
                    adata.layers["scvi_normalized"] = np.asarray(norm.values, dtype=np.float32)
                except Exception:
                    adata.layers["scvi_normalized"] = np.asarray(norm, dtype=np.float32)

                st.success("✅ SCVI trained. Latent → `obsm['X_scvi']`, normalized → `layers['scvi_normalized']`.")
        except Exception as e:
            st.error(f"SCVI failed to run: {e}")
            st.info("Tip: pin Python 3.11 in runtime.txt and add `scvi-tools` to requirements.")

    # ---------- Highly Variable Genes ----------
    st.subheader("Highly Variable Genes")

    flavor_choice = st.selectbox(
        "HVG flavor",
        ["cell_ranger", "seurat", "seurat_v3 (needs numba)"],
        key="hvg_flavor_choice",
    )
    flavor_map = {
        "cell_ranger": "cell_ranger",
        "seurat": "seurat",
        "seurat_v3 (needs numba)": "seurat_v3",
    }
    flavor = flavor_map[flavor_choice]

    n_top = st.number_input("n_top_genes", value=2000, step=500, key="hvg_n_top")

    if st.button("Find HVGs", key="find_hvg"):
        import scipy.sparse as sp

        # Work on a copy; use SCVI-normalized values if present
        ad_work = adata.copy()
        if "scvi_normalized" in ad_work.layers:
            ad_work.X = ad_work.layers["scvi_normalized"]

        # ---- Prefilter all-zero genes to avoid duplicate bin edges
        X = ad_work.X
        gsum = (np.asarray(X.sum(axis=0)).ravel() if sp.issparse(X) else X.sum(axis=0))
        ad_work_nz = ad_work[:, gsum > 0].copy()

        def _run_hvg(fl: str):
            sc.pp.highly_variable_genes(ad_work_nz, flavor=fl, n_top_genes=int(n_top))

        try:
            _run_hvg(flavor)
        except Exception as e:
            st.warning(
                f"HVG ({flavor}) failed: {e}\n"
                "→ Prefiltered zero-count genes and falling back to flavor='cell_ranger'."
            )
            _run_hvg("cell_ranger")

        # ---- Map HVG mask back to the full object (including genes dropped above)
        hv_mask = pd.Series(False, index=adata.var_names)
        hv_mask.loc[ad_work_nz.var_names] = ad_work_nz.var["highly_variable"].values
        adata.var["highly_variable"] = hv_mask.values

        # ---- Preview
        hvgs = adata.var[adata.var["highly_variable"]].copy()
        st.success(f"Found {hvgs.shape[0]} highly variable genes.")

        preview_cols = [
            c for c in ["means", "dispersions", "dispersions_norm", "variance", "variance_norm"]
            if c in ad_work_nz.var.columns
        ]
        show_cols = preview_cols[:2] if preview_cols else []
        top = ad_work_nz.var.loc[ad_work_nz.var["highly_variable"]].head(20)
        to_show = top[show_cols].copy() if show_cols else top.copy()
        to_show.insert(0, "gene", to_show.index)
        st.dataframe(to_show, width="stretch")

        # Download full HVG gene list
        csv_bytes = hvgs.index.to_series().to_csv(index=False).encode()
        st.download_button(
            "Download HVG gene list (CSV)",
            csv_bytes,
            file_name="highly_variable_genes.csv",
            mime="text/csv",
            key="hvg_download",
        )

    if st.button("Save and continue", key="normalize_save"):
        st.session_state.adata = adata
        st.success("Saved updates to session.")

# ---------------------------
# Step 4: Embedding & Clustering
# ---------------------------
if step == "Embedding & Clustering":
    if st.session_state.adata is None:
        st.stop()
    import scanpy as sc
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans

    adata = st.session_state.adata.copy()
    st.subheader("Embedding (PCA/UMAP) + Clustering")

    # -------- Embedding controls --------
    use_hvg = st.checkbox("Use only HVGs", value=True, key="ec_use_hvg")
    n_pcs = st.slider("Number of PCs", 10, 100, 50, key="ec_n_pcs")
    neighbors_k = st.slider("Neighbors k", 5, 50, 15, key="ec_neighbors_k")
    use_scvi = "X_scvi" in adata.obsm
    if use_scvi:
        st.info("SCVI latent found. If you run embedding, UMAP will be built on SCVI latent.")

    # -------- Clustering controls --------
    algo = st.selectbox(
        "Clustering algorithm",
        ["leiden (needs igraph)", "louvain (needs igraph)", "KMeans (no extra deps)"],
        index=2,
        key="ec_algo",
    )
    resolution = st.slider(
        "Resolution (graph methods; Leiden/Louvain only)", 0.1, 2.0, 0.5, 0.1, key="ec_resolution"
    )
    k_kmeans = st.slider("K (for KMeans)", 2, 50, 10, key="ec_k_kmeans")

    # -------- Action buttons --------
    colA, colB = st.columns(2)
    run_embed = colA.button("Run Embedding (PCA/UMAP)", key="ec_run_embed")
    run_cluster = colB.button("Run Clustering", key="ec_run_cluster")

    # -------- Run embedding --------
    if run_embed:
        if use_scvi:
            st.info("Using SCVI latent for neighbors/UMAP.")
            sc.pp.neighbors(adata, use_rep="X_scvi", n_neighbors=int(neighbors_k))
            sc.tl.umap(adata)
        else:
            # Work on a temporary object if HVGs requested, then copy results back
            adata_use = adata
            if use_hvg and "highly_variable" in adata.var.columns:
                adata_use = adata[:, adata.var["highly_variable"]].copy()

            sc.pp.scale(adata_use, max_value=10)
            sc.tl.pca(adata_use, n_comps=int(n_pcs))
            sc.pp.neighbors(adata_use, n_neighbors=int(neighbors_k), n_pcs=int(n_pcs))
            sc.tl.umap(adata_use)

            # Copy embeddings/graphs back to the full object
            for k in ["X_pca", "X_umap"]:
                if k in adata_use.obsm:
                    adata.obsm[k] = adata_use.obsm[k]
            for k in ["distances", "connectivities"]:
                if k in adata_use.obsp:
                    adata.obsp[k] = adata_use.obsp[k]

        st.session_state.adata = adata
        st.success("Computed neighbors and UMAP.")

    # -------- Run clustering --------
    if run_cluster:
        try:
            if algo.startswith("leiden"):
                sc.tl.leiden(adata, resolution=float(resolution))
                st.success("Leiden clustering done.")
            elif algo.startswith("louvain"):
                sc.tl.louvain(adata, resolution=float(resolution))
                st.success("Louvain clustering done.")
            else:
                # KMeans uses PCA; compute quickly if missing
                if "X_pca" not in adata.obsm:
                    st.info("PCA not found; computing PCA (50 comps) quickly for KMeans…")
                    sc.pp.scale(adata, max_value=10)
                    sc.tl.pca(adata, n_comps=50)
                X = adata.obsm["X_pca"]
                labels = KMeans(n_clusters=int(k_kmeans), n_init=10, random_state=0).fit_predict(X)
                adata.obs["kmeans"] = pd.Categorical(labels.astype(str))
                st.success("KMeans clustering done.")
        except ImportError as e:
            # igraph/leidenalg/louvain missing → fallback to KMeans
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

    # -------- Plotting --------
    # Build a color list (clusters first)
    cluster_keys = [c for c in ["leiden", "louvain", "kmeans"] if c in adata.obs.columns]
    candidate_cols = cluster_keys.copy()
    # then small-cardinality categorical columns
    for c in adata.obs.columns:
        if c in candidate_cols:
            continue
        s = adata.obs[c]
        if pd.api.types.is_categorical_dtype(s) or s.dtype == object:
            if s.nunique() <= 50:
                candidate_cols.append(c)

    color_by = st.selectbox(
        "Color UMAP by",
        options=["(none)"] + candidate_cols,
        index=1 if candidate_cols else 0,  # default to first cluster key if available
        key="ec_color_by",
    )
    color_key = None if color_by == "(none)" else color_by

    if "X_umap" in adata.obsm:
        fig = _umap_scatter(adata, color_key=color_key)
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )
        if color_key in cluster_keys:
            st.dataframe(
                adata.obs[color_key].value_counts().rename_axis("cluster").reset_index(name="n"),
                width="stretch",
            )
    else:
        st.warning("UMAP not computed yet. Click “Run Embedding (PCA/UMAP)”.")

# ---------------------------
# Step 6: Differential Expression
# ---------------------------
if step == "Markers & DE":
    if st.session_state.adata is None:
        st.stop()
    import scanpy as sc
    import pandas as pd
    import numpy as np

    adata = st.session_state.adata.copy()
    st.subheader("Differential Expression (rank_genes_groups)")

    # ---- choose groupby (categorical or convertible, 2..50 groups)
    candidate_groupbys = []
    for c in adata.obs.columns:
        s = adata.obs[c]
        if pd.api.types.is_categorical_dtype(s):
            if 2 <= len(s.cat.categories) <= 50:
                candidate_groupbys.append(c)
        elif s.dtype == object or pd.api.types.is_integer_dtype(s):
            nunq = s.nunique(dropna=True)
            if 2 <= nunq <= 50:
                candidate_groupbys.append(c)
    for k in ["leiden", "louvain", "kmeans"]:
        if k in candidate_groupbys:
            candidate_groupbys.remove(k)
            candidate_groupbys.insert(0, k)

    if not candidate_groupbys:
        st.warning("No suitable group columns in .obs (need 2–50 groups). Run clustering first.")
        st.stop()

    groupby = st.selectbox("Group by", candidate_groupbys, key="de_groupby")
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        adata.obs[groupby] = adata.obs[groupby].astype("category")

    # ---- filter tiny groups
    min_cells = st.number_input("Minimum cells per group", 2, 100, 5, key="de_min_cells")
    counts = adata.obs[groupby].value_counts().sort_index()
    valid_groups = counts[counts >= int(min_cells)].index.tolist()
    tiny_groups = counts[counts < int(min_cells)]

    if len(valid_groups) < 2:
        st.error(
            f"Not enough groups after filtering (need ≥2). "
            f"Group sizes:\n{counts.to_dict()}"
        )
        st.info("Lower clustering resolution, merge small clusters, or reduce the minimum.")
        st.stop()

    if len(tiny_groups) > 0:
        st.info(f"Excluding tiny groups (<{int(min_cells)} cells): {tiny_groups.to_dict()}")

    mask = adata.obs[groupby].isin(valid_groups)
    adata_use = adata[mask].copy()
    adata_use.obs[groupby] = adata_use.obs[groupby].cat.remove_unused_categories()

    # ---- reference & method
    cats = list(adata_use.obs[groupby].cat.categories)
    ref_choice = st.selectbox("Reference group", ["rest"] + cats, key="de_reference")
    reference = "rest" if ref_choice == "rest" else ref_choice

    method = st.selectbox(
        "Test method",
        ["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
        index=0,
        key="de_method",
    )

    # ---- expression layer
    layer_opts = ["X"]
    if "scvi_normalized" in adata.layers:
        layer_opts.append("scvi_normalized")
    layer_choice = st.selectbox("Expression layer", layer_opts, key="de_layer")
    layer = None if layer_choice == "X" else layer_choice

    # ---- optional HVG restriction for speed
    use_hvg = st.checkbox("Limit to HVGs (if present)", value=True, key="de_use_hvg")
    gene_mask = adata_use.var["highly_variable"].values if (use_hvg and "highly_variable" in adata_use.var.columns) else None
    if gene_mask is not None:
        adata_use = adata_use[:, gene_mask].copy()

    if st.button("Run differential expression", key="de_run"):
        try:
            sc.tl.rank_genes_groups(
                adata_use,
                groupby=groupby,
                method=method,
                layer=layer,
                reference=reference,
            )
            df = sc.get.rank_genes_groups_df(adata_use, group=None)
            st.success("Computed DE with rank_genes_groups.")
            st.dataframe(df.head(50), width="stretch")
            st.download_button(
                "Download full DE results (CSV)",
                df.to_csv(index=False).encode(),
                file_name=f"rank_genes_groups_{groupby}_{method}.csv",
                mime="text/csv",
                key="de_download",
            )
            # persist results back
            adata.uns["rank_genes_groups"] = adata_use.uns["rank_genes_groups"]
            if "rank_genes_groups_params" in adata_use.uns:
                adata.uns["rank_genes_groups_params"] = adata_use.uns["rank_genes_groups_params"]
            st.session_state.adata = adata
        except Exception as e:
            st.error(f"DE failed: {e}")

# ---------------------------
# Step 7: Cell Type Annotation
# ---------------------------
if step == "Cell Type Annotation":
    if st.session_state.adata is None:
        st.stop()

    import scanpy as sc
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import urllib.request

    st.subheader("Cell Type Annotation (optional)")
    st.markdown(
        """
        This step uses **pretrained CellTypist models** (and optionally other tools)
        to automatically assign cell-type identities based on transcriptomic profiles.

        You can choose between **Human** and **Mouse** models and run annotation directly.
        """
    )

    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)

	    # ---------- Robust model fetcher ----------
        # ---------- Robust model fetcher (FINAL) ----------
    def ensure_celltypist_model(model_filename: str) -> Path | None:
        """
        Try to make sure `data/models/<model_filename>` exists.
        Order:
          1) CellTypist's official downloader (try both API signatures)
          2) Two static mirrors (may fail if DNS/egress blocked)
          3) Return None (caller can ask for manual upload)
        """
        target = model_dir / model_filename
        if target.exists():
            return target

        # 1) Preferred: CellTypist's downloader (handles mirrors/cache)
        try:
            import celltypist
            from pathlib import Path as _Path
            # Some versions expose these in different places; try both.
            try:
                from celltypist.models import download_models, models_path
            except ImportError:
                from celltypist import models as _models
                download_models = _models.download_models
                models_path = _models.models_path

            with st.spinner(f"Downloading {model_filename} via CellTypist…"):
                stem = model_filename.replace(".pkl", "")
                # Try both call signatures to be version-agnostic
                try:
                    download_models(model=stem)          # older/newer signature
                except TypeError:
                    download_models(models=[stem])       # alternative signature

                ct_dir = _Path(models_path())
                src = ct_dir / f"{stem}.pkl"
                if src.exists():
                    target.write_bytes(src.read_bytes())
                    st.success(f"✅ Downloaded to {target}")
                    return target
                else:
                    st.warning(f"CellTypist reported success but {src} not found.")
        except (ModuleNotFoundError, ImportError):
            st.error("No module named `celltypist`. Add `celltypist>=1.6.0` to requirements and redeploy.")
        except Exception as e:
            st.warning(f"CellTypist downloader failed: {e}")

        # 2) Fallback: static URLs (will fail if DNS/egress blocked)
        import urllib.request
        for base in [
            "https://celltypist.cellgeni.sanger.ac.uk/models/",
            "https://celltypist.sanger.ac.uk/models/",
        ]:
            try:
                url = base + model_filename
                with st.spinner(f"Downloading {model_filename} from {base} …"):
                    urllib.request.urlretrieve(url, target)
                st.success(f"✅ Downloaded to {target}")
                return target
            except Exception as e:
                st.info(f"Fallback URL failed: {e}")

        # 3) Give up (let caller prompt for manual upload)
        return None


    # -------------------------------------------------
    # Organism selection
    # -------------------------------------------------
    organism = st.selectbox("Organism", ["Human", "Mouse"], key="ct_organism")

    MODEL_CATALOG = {
        "Human": {
			"Immune (low-res)": "Immune_All_Low.pkl",
	        "Immune (high-res)": "Immune_All_High.pkl",
	        "All tissues (pan-fetal)": "Pan_Fetal_Human.pkl",
	        "Lung (atlas)": "Human_Lung_Atlas.pkl",
	        "PBMC (blood)": "Healthy_COVID19_PBMC.pkl",
        },
        "Mouse": {
			"All tissues (liver ref)": "Healthy_Mouse_Liver.pkl",
	        "Immune (developing brain)": "Developing_Mouse_Brain.pkl",
	        "Brain (whole)": "Mouse_Whole_Brain.pkl",
        },
    }
    organism_models = MODEL_CATALOG[organism]

    model_choice = st.selectbox(
        f"Choose {organism} model",
        list(organism_models.keys()) + ["Custom (.pkl path)", "Upload .pkl now"],
        key="ct_model_choice",
    )

    uploaded_pkl = None
    model_path = None

    if model_choice == "Custom (.pkl path)":
        custom_path = st.text_input(
            "Enter full path to your .pkl model file",
            value=str(model_dir / "Your_Model.pkl"),
            key="ct_custom_path",
        )
        model_path = Path(custom_path)

    elif model_choice == "Upload .pkl now":
        uploaded_pkl = st.file_uploader("Upload CellTypist model (.pkl)", type=["pkl"], key="ct_upload")
        if uploaded_pkl is not None:
            target = model_dir / uploaded_pkl.name
            target.write_bytes(uploaded_pkl.read())
            st.success(f"✅ Saved uploaded model to {target}")
            model_path = target

    else:
        model_filename = organism_models[model_choice]
        candidate = ensure_celltypist_model(model_filename)
        if candidate is None:
            st.error(
                "Couldn't download the model (network/DNS blocked). "
                "Please switch to **Upload .pkl now** above and provide the model file, "
                "or commit it under `data/models/` in your repo."
            )
        else:
            model_path = candidate

    # -------------------------------------------------
    # Run CellTypist
    # -------------------------------------------------
    can_run = model_path is not None and Path(model_path).exists()
    if st.button("Run CellTypist", key="celltypist_run"):
        if not can_run:
            st.error("No model file available. Please upload or provide a valid `.pkl` path.")
        else:
            try:
                import celltypist
                adata = st.session_state.adata.copy()
                with st.spinner(f"Annotating cells using {Path(model_path).name}…"):
                    pred = celltypist.annotate(adata, model=str(model_path))
                    adata.obs["celltypist_label"] = pred.predicted_labels
                st.session_state.adata = adata
                st.success("✅ CellTypist annotation complete.")
                st.dataframe(
                    adata.obs["celltypist_label"].value_counts()
                    .rename_axis("Cell Type").reset_index(name="n"),
                    width="stretch",
                )
                if "X_umap" in adata.obsm:
                    fig = _umap_scatter(adata, color_key="celltypist_label")
                    if fig is not None:
                        st.plotly_chart(fig, width="stretch",
                                        config={"displaylogo": False, "responsive": True})
                else:
                    st.info("No UMAP found. Compute embedding first to visualize labels.")
            except ModuleNotFoundError:
                st.error("`celltypist` is missing. Add `celltypist>=1.6.0` to requirements.txt and redeploy.")
            except Exception as e:
                st.error(f"CellTypist failed: {e}")

    st.divider()

    # -------------------------------------------------
    # Future: other annotation tools
    # -------------------------------------------------
    st.markdown("### 🧩 Other annotation options (coming soon)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Azimuth (via Seurat reference)", key="azimuth_placeholder"):
            st.info("Azimuth integration planned — will require R or REST API backend.")
    with col2:
        if st.button("SingleR (reference-based in R)", key="singler_placeholder"):
            st.info("SingleR integration planned — via Bioconductor REST service.")

    st.caption("You can add your own pretrained CellTypist models under `data/models/`.")
    if st.button("Save and continue", key="celltypist_save"):
        st.success("Saved annotations to session.")

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
