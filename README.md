# Single-Cell RNA-seq Streamlit App (Demo-Friendly)

Interactive scRNA-seq analysis pipeline: **QC → normalization → DR → clustering → markers**, with optional CellTypist and RNA velocity.

**Demo cap**: Limits cells (default 5,000) for safe use on Streamlit Cloud. Toggle in sidebar.

---

## 🚀 Streamlit Cloud

1. Push this repo to GitHub with `app.py`, `requirements.txt`, and `runtime.txt`.
2. Deploy to Streamlit Cloud with entrypoint `app.py`.
3. Upload a small `.h5ad` (<100 MB).

## 💻 Local run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# Optional extras:
# pip install -r optional-requirements.txt
streamlit run app.py
```

## 🧩 Features
- Load `.h5ad`, 10x MTX, or 10x HDF5.
- QC & filtering.
- Normalization (`normalize_total` + `log1p`).
- PCA → UMAP → clustering.
- Marker discovery & export.
- Optional: CellTypist + scVelo preview.

## 🧠 Notes
- For big data, keep the demo cap enabled or deploy on your own VM.
- Optional extras in `optional-requirements.txt`.

---

MIT License
