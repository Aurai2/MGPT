
# MaterialsGPT – Streamlit GUI

This converts your notebook workflow into a web app:

- **Phase‑1**: text → embedding → BIRCH micro‑clusters → Agglomerative macro‑clusters → engineered features → `phase1_features.parquet`  
- **Phase‑2**: merge (by `row_id` or index) → train **RandomForest** and **ElasticNet** → report R² & MAE on 70/20/10 split

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload your CSV (must include a text column like `process`). If you have an ID column, name it `row_id` (or change in the sidebar).

### Notes
- Embedding models download on first run (requires internet).
- If you lack a shared key between CSV and features, keep row order unchanged between Phase‑1 and Phase‑2.
- You can change composition column prefix (default `pct_`) or pick targets explicitly in the UI.
