"""Shared data loading, CV splitting and scoring."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

HERE = Path(__file__).resolve().parent
DROP = ["y", "author_id", "create_date", "music_id"]
SEED = 42


def load(name=None):
    import os
    name = name or os.environ.get("FEATS", "features2.pkl")
    p = HERE / "data" / name
    if not p.exists():
        p = HERE / "data" / "features.pkl"
    df = pd.read_pickle(p)
    y = df["y"].to_numpy(dtype=np.float64)
    groups = df["author_id"].to_numpy()
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    for c in X.columns:
        if str(X[c].dtype) == "object":
            X[c] = X[c].astype("category")
    return X, y, groups, df


def splits(y, groups, mode="random", n=5):
    if mode == "group":
        return list(GroupKFold(n_splits=n).split(y, y, groups))
    return list(KFold(n_splits=n, shuffle=True, random_state=SEED).split(y))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
