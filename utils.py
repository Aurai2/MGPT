
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def load_csv_safely(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_feature_columns(df: pd.DataFrame, composition_prefix: str | None) -> list[str]:
    # Use numeric columns + optional composition prefix
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if composition_prefix:
        comp_cols = [c for c in df.columns if str(c).startswith(composition_prefix)]
    else:
        comp_cols = []
    # Deduplicate but preserve order
    seen = set()
    cols = []
    for c in comp_cols + numeric_cols:
        if c not in seen:
            seen.add(c)
            cols.append(c)
    return cols

def align_and_merge_features(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    target_cols: list[str] | None,
    id_col: str | None,
    composition_prefix: str | None
):
    # Build a key for join
    if id_col and id_col in df.columns and "key" in feat.columns:
        df = df.copy()
        df["key"] = df[id_col].astype(str)
        merged = df.merge(feat, on="key", how="inner", validate="many_to_one")
    else:
        # fallback to index alignment length-wise
        df = df.copy().reset_index(drop=True)
        feat = feat.copy().reset_index(drop=True)
        if len(df) != len(feat):
            raise ValueError("Row count mismatch and no shared key to merge on. Provide an ID column.")
        merged = pd.concat([df, feat.drop(columns=["key"], errors="ignore")], axis=1)

    # Select features
    base_feats = [c for c in merged.columns if c.startswith("cluster_") or c.startswith("proc_pca_") or c.startswith("route_") or c.startswith("form_")]
    comp_feats = pick_feature_columns(merged, composition_prefix)
    X_cols = sorted(set(base_feats + comp_feats))
    X = merged[X_cols].select_dtypes(include=["number"]).fillna(0.0).to_numpy()

    # Targets
    if target_cols:
        used_targets = [c for c in target_cols if c in merged.columns]
    else:
        # heuristic default targets
        candidates = ["UTS", "YS", "Elongation", "Hardness", "Tensile", "Yield"]
        used_targets = [c for c in candidates if c in merged.columns]
        if not used_targets:
            # last resort: pick 1-3 numeric targets at the end
            tail_numeric = [c for c in merged.columns if c not in X_cols and pd.api.types.is_numeric_dtype(merged[c])][-3:]
            used_targets = tail_numeric

    if not used_targets:
        raise ValueError("No valid target columns found. Please select targets in the UI.")

    Y = merged[used_targets].to_numpy().astype(float)

    return X, Y, used_targets
