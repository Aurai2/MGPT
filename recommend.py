
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def _standardize(df: pd.DataFrame, cols: List[str]):
    """Return standardized array, plus (mean, std) for each column."""
    X = df[cols].astype(float).to_numpy()
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    Z = (X - means) / stds
    return Z, means, stds

def recommend_alloys(
    df: pd.DataFrame,
    property_cols: List[str],
    targets: Dict[str, float],
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    include_keyword_col: Optional[str] = None,
    include_keyword: Optional[str] = None,
    exclude_keyword: Optional[str] = None,
    extra_sort_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Recommend alloys whose measured properties are closest to desired targets.
    - property_cols: numeric columns to compare (e.g., ["UTS","YS","Elongation"])
    - targets: mapping of property -> target value
    - weights: optional mapping of property -> nonnegative weight (defaults to 1.0)
    - include/exclude keyword filter applied to include_keyword_col (e.g., 'process')
    """
    if not property_cols:
        raise ValueError("No property columns provided for recommendation.")
    # Filter rows with complete data for required properties
    work = df.dropna(subset=property_cols).copy()
    if work.empty:
        return work

    # Optional keyword filters
    if include_keyword_col and include_keyword:
        mask = work[include_keyword_col].astype(str).str.contains(include_keyword, case=False, na=False)
        work = work[mask]
    if include_keyword_col and exclude_keyword:
        mask = ~work[include_keyword_col].astype(str).str.contains(exclude_keyword, case=False, na=False)
        work = work[mask]

    if work.empty:
        return work

    # Standardize properties to avoid scale issues
    Z, means, stds = _standardize(work, property_cols)

    # Build target vector (standardized using same stats)
    tgt = np.array([targets.get(c, work[c].astype(float).mean()) for c in property_cols], dtype=float)
    tgtZ = (tgt - means) / stds

    # Weights
    w = np.array([weights.get(c, 1.0) if weights else 1.0 for c in property_cols], dtype=float)
    w = np.maximum(w, 0.0)
    if (w > 0).sum() == 0:
        w[:] = 1.0

    # Weighted Euclidean distance in standardized space
    diff = Z - tgtZ
    dist = np.sqrt((diff**2 * w).sum(axis=1))

    out = work.copy()
    out.insert(0, "rec_distance", dist)
    out = out.sort_values("rec_distance", ascending=True)

    if extra_sort_cols:
        out = out.sort_values(extra_sort_cols + ["rec_distance"], ascending=[False] * len(extra_sort_cols) + [True])

    return out.head(top_k).reset_index(drop=True)
