
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

def _hash_texts(texts: pd.Series) -> str:
    import hashlib
    h = hashlib.md5()
    for t in texts.astype(str).tolist():
        h.update(t.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def build_embeddings_cache(
    df: pd.DataFrame,
    text_col: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[np.ndarray, list[str], str]:
    """
    Returns (embeddings, keys, cache_key) for the dataframe.
    Cache key depends on text content + model name.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column `{text_col}` not found in data.")
    texts = df[text_col].fillna("").astype(str).str.strip()
    cache_key = f"{model_name}|{_hash_texts(texts)}"

    model = SentenceTransformer(model_name)
    X = model.encode(texts.tolist(), show_progress_bar=False, convert_to_numpy=True)

    # Prefer an identifier column if present
    if "row_id" in df.columns:
        keys = df["row_id"].astype(str).tolist()
    elif "key" in df.columns:
        keys = df["key"].astype(str).tolist()
    else:
        keys = [str(i) for i in range(len(df))]

    return X, keys, cache_key

def semantic_search(
    df: pd.DataFrame,
    text_col: str,
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 25
) -> pd.DataFrame:
    """
    Run cosine-nearest-neighbor search over `text_col` using SentenceTransformer embeddings.
    Returns a new DataFrame with a `similarity` column (0..1, higher is closer).
    """
    if not query or not query.strip():
        return df.head(0)

    X, keys, cache_key = build_embeddings_cache(df, text_col, model_name)
    model = SentenceTransformer(model_name)
    qv = model.encode([query.strip()], show_progress_bar=False, convert_to_numpy=True)

    nn = NearestNeighbors(
        n_neighbors=min(top_k, len(X)),
        metric="cosine"
    )
    nn.fit(X)
    dist, idx = nn.kneighbors(qv)
    idx = idx[0]
    dist = dist[0]
    # cosine distance → similarity
    sim = 1.0 - dist

    results = df.iloc[idx].copy()
    results.insert(0, "similarity", sim.round(4))
    return results
