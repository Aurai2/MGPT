
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from phase1 import export_phase1_features
from phase2_rf import train_random_forest
from phase2_enet import train_elasticnet

def prepare_backend(
    csv_path: str,
    process_col: str = "process",
    id_col: Optional[str] = "row_id",
    features_out: str = "phase1_features.parquet",
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    birch_threshold: float = 0.25,
    agglo_clusters: int = 12,
    composition_prefix: Optional[str] = "pct_",
    target_cols: Optional[List[str]] = None,
    rf_params: Optional[dict] = None,
    enet_params: Optional[dict] = None,
) -> dict:
    """
    Runs Phase‑1 (always), then trains RF and ENet (optional but default on), and
    returns a dict with:
      - df: the original dataframe (with minimal cleaning)
      - features_path: saved Phase‑1 features file
      - targets: the resolved list of target columns used for Phase‑2
      - rf_results / enet_results: metrics and fitted params (not the heavy model object)
    """
    df = pd.read_csv(csv_path)
    # Phase‑1
    features_path = export_phase1_features(
        csv_path=csv_path,
        process_col=process_col,
        id_col=id_col if (id_col and id_col in df.columns) else None,
        out_path=features_out,
        embed_model=embed_model,
        birch_threshold=birch_threshold,
        agglo_clusters=agglo_clusters
    )

    # Phase‑2 defaults
    if target_cols is None:
        # heuristic default based on common property names
        candidates = ["UTS", "YS", "Elongation", "Hardness", "Tensile", "Yield"]
        target_cols = [c for c in candidates if c in df.columns]
        if not target_cols:
            # fallback to last 2-3 numeric columns
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            target_cols = num_cols[-3:]

    rf_params = rf_params or {"n_estimators": 300, "max_depth": None, "random_state": 42}
    enet_params = enet_params or {"alpha": 1e-3, "l1_ratio": 0.5, "random_state": 42}

    # Train RF & ENet
    rf_results = train_random_forest(
        csv_path=csv_path,
        features_path=features_path,
        target_cols=target_cols,
        process_col=process_col,
        id_col=id_col if (id_col and id_col in df.columns) else None,
        composition_prefix=composition_prefix,
        **rf_params
    )
    enet_results = train_elasticnet(
        csv_path=csv_path,
        features_path=features_path,
        target_cols=target_cols,
        process_col=process_col,
        id_col=id_col if (id_col and id_col in df.columns) else None,
        composition_prefix=composition_prefix,
        **enet_params
    )

    return {
        "df_head": df.head(5).to_dict(orient="list"),
        "features_path": features_path,
        "targets": target_cols,
        "rf_results": rf_results,
        "enet_results": enet_results,
    }
