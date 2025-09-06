
from pathlib import Path
import pandas as pd
import numpy as np

# Embeddings / clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

def _clean_texts(series):
    return series.fillna("").astype(str).str.strip()

def export_phase1_features(
    csv_path: str,
    process_col: str = "process",
    id_col: str | None = "row_id",
    out_path: str = "phase1_features.parquet",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    birch_threshold: float = 0.25,
    agglo_clusters: int = 12
) -> str:
    """
    Loads the dataset, builds text embeddings from `process_col`, clusters with BIRCH+Agglomerative,
    adds light engineered features, and saves a row-keyed feature table for Phase 2.
    """
    df = pd.read_csv(csv_path)
    if process_col not in df.columns:
        raise ValueError(f"Column `{process_col}` not in CSV. Found: {list(df.columns)}")

    # Derive a stable key
    if id_col and id_col in df.columns:
        key = df[id_col].astype(str)
    else:
        key = df.reset_index().index.astype(str)

    texts = _clean_texts(df[process_col])
    if (texts == "").all():
        raise ValueError("All processing texts are empty after cleaning.")

    model = SentenceTransformer(embed_model)
    X = model.encode(texts.tolist(), show_progress_bar=False, convert_to_numpy=True)

    # Micro-clusters with BIRCH
    birch = Birch(n_clusters=None, threshold=birch_threshold)
    micro_labels = birch.fit_predict(X)
    centers = birch.subcluster_centers_

    # Macro clusters on micro centers
    agglo = AgglomerativeClustering(n_clusters=agglo_clusters)
    macro_on_centers = agglo.fit_predict(centers)
    # Map each sample to its macro label via its micro center assignment
    macro_labels = macro_on_centers[micro_labels]

    # PCA of embeddings (small dimensional signal)
    pca = PCA(n_components=min(8, X.shape[1]))
    X_pca = pca.fit_transform(X)

    # One-hot of macro labels
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    macro_oh = enc.fit_transform(macro_labels.reshape(-1, 1))
    oh_cols = [f"cluster_{i}" for i in range(macro_oh.shape[1])]

    # Build feature frame
    feat = pd.DataFrame({
        "key": key,
        "macro_cluster": macro_labels.astype(int)
    })
    for i, col in enumerate(oh_cols):
        feat[col] = macro_oh[:, i]

    # Add PCA columns
    for j in range(X_pca.shape[1]):
        feat[f"proc_pca_{j+1}"] = X_pca[:, j]

    # Example lightweight flags (route/form) via simple keyword rules; adjust as needed
    lower_texts = texts.str.lower()
    feat["route_heat_treat"] = lower_texts.str.contains("quench|temper|anneal|normalize").astype(int)
    feat["route_surface"] = lower_texts.str.contains("carbur|nitrid|boroniz|shot peen").astype(int)
    feat["form_plate"] = lower_texts.str.contains("plate|sheet").astype(int)
    feat["form_bar"] = lower_texts.str.contains("bar|rod").astype(int)

    # Save
    out = Path(out_path)
    feat.to_parquet(out, index=False)
    return str(out)
