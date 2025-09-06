
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from utils import load_csv_safely, align_and_merge_features

def train_elasticnet(
    csv_path: str,
    features_path: str = "phase1_features.parquet",
    target_cols: list[str] | None = None,
    process_col: str = "process",
    id_col: str | None = "row_id",
    composition_prefix: str | None = "pct_",
    alpha: float = 1e-3,
    l1_ratio: float = 0.5,
    random_state: int = 42
) -> dict:
    df = load_csv_safely(csv_path)
    feat = pd.read_parquet(features_path)

    X, y, used_targets = align_and_merge_features(
        df, feat, target_cols=target_cols, id_col=id_col, composition_prefix=composition_prefix
    )

    # 70/20/10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1/3, random_state=random_state
    )

    base = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state, max_iter=20000)
    pipe = make_pipeline(StandardScaler(with_mean=False), base)

    if y.shape[1] > 1:
        model = MultiOutputRegressor(pipe, n_jobs=-1)
    else:
        model = pipe

    model.fit(X_train, y_train)

    def evaluate(splitX, splity):
        pred = model.predict(splitX)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        r2 = float(np.mean([r2_score(splity[:, i], pred[:, i]) for i in range(splity.shape[1])]))
        mae = float(np.mean([np.abs(splity[:, i] - pred[:, i]).mean() for i in range(splity.shape[1])]))
        return r2, mae

    r2_val, mae_val = evaluate(X_val, y_val)
    r2_test, mae_test = evaluate(X_test, y_test)

    return {
        "targets": used_targets,
        "model": "ElasticNet",
        "params": {"alpha": alpha, "l1_ratio": l1_ratio, "random_state": random_state},
        "val_r2": r2_val, "val_mae": mae_val,
        "test_r2": r2_test, "test_mae": mae_test,
        "train_size": int(X_train.shape[0]), "val_size": int(X_val.shape[0]), "test_size": int(X_test.shape[0])
    }
