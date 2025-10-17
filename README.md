# 🧬 Single-Cell RNA-seq Analysis Portal

An interactive **Streamlit-based app** for end-to-end single-cell RNA-seq (scRNA-seq) analysis — from raw count matrices to annotated cell clusters — powered by **Scanpy**, **scvi-tools**, and **CellTypist**.

---

## 🚀 Features

### 🔍 Data Input
- Upload `.h5`, `.h5ad`, or 10X Genomics filtered feature matrix files.
- Automatic recognition of `.h5` and `.h5ad` file structures.
- Live preview of cell × gene count matrices.

### 🧪 Preprocessing & QC
- Compute per-cell and per-gene QC metrics.
- Filter low-quality cells/genes interactively.

### ⚙️ Normalization & HVG Detection
- `sc.pp.normalize_total + log1p` standard normalization.
- **SCVI**-based variance stabilization (optional, GPU/CPU).
- Robust HVG computation (cellranger, seurat, seurat_v3).

### 🌀 Dimensionality Reduction & Clustering
- PCA + UMAP visualization.
- Leiden, Louvain, and KMeans clustering.
- Adjustable resolution and neighbors.

### 🧠 Cell Type Annotation
- Uses **CellTypist pretrained models** for automatic annotation.
- Supports both **Human** and **Mouse** models.
- Allows direct upload of custom `.pkl` models.
- Includes data sanitization and normalization before annotation.

### 📊 Visualization
- Interactive **Plotly UMAPs** with color-by-cluster or cell-type.
- Live count tables and download options.

### 💾 Export
- Download lists of highly variable genes (CSV).
- Save intermediate `.h5ad` AnnData for downstream work.

---

## 🧱 Tech Stack

| Component | Library |
|------------|----------|
| UI | [Streamlit](https://streamlit.io) |
| Core analysis | [Scanpy](https://scanpy.readthedocs.io/en/stable/) |
| Variance modeling | [scvi-tools](https://scvi-tools.org) |
| Annotation | [CellTypist](https://www.celltypist.org) |
| Visualization | [Plotly](https://plotly.com/python/) |
| ML | scikit-learn, NumPy, pandas |

---
Traag et al., 2019 — Leiden algorithm
Gayoso et al., 2022 — scvi-tools for probabilistic modeling
Domínguez Conde et al., 2022 — CellTypist: automated cell-type annotation for scRNA-seq

🧑‍💻 Authors

Developed by Dr. Moganti  and contributors.
email: kmoganti1@gmail.com
Built for scalable, transparent, and reproducible single-cell analysis.


## 🧩 Installation

```bash
# Clone this repository
git clone https://github.com/<your-username>/singlecellsequencing.git
cd singlecellsequencing

# Create and activate environment
conda create -n scrna python=3.11 -y
conda activate scrna

# Install dependencies
pip install -r requirements.txt
