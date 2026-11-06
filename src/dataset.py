"""Buoc 3: dung ma tran dac trung dense (float32) va nap lai bang memmap.

Ma tran duoc ghi ra artifacts/X.npy de khong chiem RAM thuong truc; khi train
se mo lai o che do mmap_mode="r" de he dieu hanh co the thu hoi trang.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy import sparse

from src import config as C

X_PATH = C.ARTIFACTS / "X.npy"
COLS_PATH = C.ARTIFACTS / "X_columns.json"
ROW_PATH = C.ARTIFACTS / "X_row_order.npy"
AGE_PATH = C.ARTIFACTS / "y_age.npy"

DROP_BASE = {"cust_int", "Active", "postal_size", "log_postal_size"}


def build() -> tuple[int, list[str]]:
    base = pl.read_parquet(C.ARTIFACTS / "customer_base_features.parquet").sort("cust_int")
    base_cols = [c for c in base.columns if c not in DROP_BASE]
    n = base.height

    meta = json.loads(C.FEATURE_META_JSON.read_text())
    blocks: list[np.ndarray] = [base.select(base_cols).to_numpy().astype(np.float32)]
    names: list[str] = list(base_cols)

    total_txn = base.get_column("n_txn").to_numpy().astype(np.float32)
    total_txn = np.where(total_txn > 0, total_txn, 1.0)

    for col in C.CAT_COLS:
        mat = sparse.load_npz(C.ARTIFACTS / f"cat_counts_{col}.npz").tocsr()
        assert mat.shape[0] == n, (col, mat.shape, n)
        share = mat.astype(np.float32).multiply(1.0 / total_txn[:, None]).toarray()
        blocks.append(share)
        names.extend(meta["cat_feature_names"][col])

    X = np.lib.format.open_memmap(X_PATH, mode="w+", dtype=np.float32, shape=(n, len(names)))
    offset = 0
    for blk in blocks:
        X[:, offset : offset + blk.shape[1]] = blk
        offset += blk.shape[1]
    X.flush()
    del X

    y = (
        pl.read_parquet(C.ARTIFACTS / "customers_base.parquet", columns=["cust_int", "age"])
        .sort("cust_int")
        .get_column("age")
        .to_numpy()
        .astype(np.float32)
    )
    np.save(AGE_PATH, y)
    np.save(ROW_PATH, base.get_column("cust_int").to_numpy().astype(np.int64))
    COLS_PATH.write_text(json.dumps(names, indent=2))
    return n, names


def load() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(AGE_PATH)
    names = json.loads(COLS_PATH.read_text())
    return X, y, names


if __name__ == "__main__":
    n, names = build()
    print(f"X: {n:,} x {len(names)} features -> {X_PATH}")
