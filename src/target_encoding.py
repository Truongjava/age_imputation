"""Target encoding an toàn theo fold cho bai toan age imputation.

Voi moi muc phan loai (article, department, product_type...), gia tri encoding cua
mot muc la tuoi trung binh (lam tron Bayes) cua cac khach hang CO NHAN trong tap
train cua fold. Dac trung cua khach hang la trung binh co trong so theo so luot mua.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

from src import config as C

ALPHA_ARTICLE = 10.0
ALPHA_LEVEL = 20.0


def _smooth(sums: np.ndarray, cnts: np.ndarray, prior: float, alpha: float) -> np.ndarray:
    return (sums + prior * alpha) / (cnts + alpha)


def _customer_feature(
    mat: sparse.csr_matrix, enc: np.ndarray, prior: float
) -> np.ndarray:
    num = np.asarray(mat @ enc).ravel()
    den = np.asarray(mat.sum(axis=1)).ravel()
    safe = np.where(den > 0, den, 1.0)
    return np.where(den > 0, num / safe, prior).astype(np.float32)


def _level_encoding(
    mat: sparse.csr_matrix,
    train_mask: np.ndarray,
    y: np.ndarray,
    prior: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    sub = mat[train_mask]
    sums = np.asarray(sub.T @ y[train_mask]).ravel()
    cnts = np.asarray(sub.sum(axis=0)).ravel()
    return _smooth(sums, cnts, prior, alpha), cnts


class TargetEncoder:
    """Tinh cac dac trung target-encoding chi tu tap train cua fold."""

    def __init__(self, n_customers: int) -> None:
        meta = pl.read_parquet(C.ARTIFACTS / "articles_levels.parquet").sort("art_int")
        self.n_customers = n_customers
        self.n_articles = meta.height
        self.cust_article = sparse.load_npz(C.ARTIFACTS / "cust_article.npz").tocsr()
        self.levels = {
            col: sparse.load_npz(C.ARTIFACTS / f"lvl_{col}.npz").tocsr()
            for col in [c for c in meta.columns if c != "art_int"]
        }
        self.names = [
            "te_article",
            "te_department_no",
            "te_product_type_no",
            "te_index_name",
            "te_section_name",
            "te_product_type_name",
            "art_enc_std",
            "art_enc_max",
            "art_enc_min",
        ]

    @property
    def n_features(self) -> int:
        return len(self.names)

    def transform(self, train_mask: np.ndarray, y: np.ndarray) -> np.ndarray:
        train_mask = train_mask & ~np.isnan(y)
        y_train = y[train_mask]
        prior = float(np.median(y_train))
        n = self.n_customers
        feats = np.empty((n, self.n_features), dtype=np.float32)

        # article: tin hieu manh nhat, lam tron nhe
        enc, _ = _level_encoding(self.cust_article, train_mask, y, prior, ALPHA_ARTICLE)
        feats[:, 0] = _customer_feature(self.cust_article, enc, prior)

        for j, col in enumerate(
            ["c_department_no", "c_product_type_no", "c_index_name", "c_section_name", "c_product_type_name"],
            start=1,
        ):
            mat = self.levels[col]
            enc_lvl, _ = _level_encoding(mat, train_mask, y, prior, ALPHA_LEVEL)
            feats[:, j] = _customer_feature(mat, enc_lvl, prior)

        # Do phan tan "do tuoi" cua cac mon da mua: khach mua mot nhom tuoi nhat quan
        # (std thap) thi du doan dang tin cay hon.
        mat = self.cust_article
        vals = enc[mat.indices]
        indptr = mat.indptr
        cnt = np.diff(indptr)
        has = cnt > 0
        mx = np.full(n, prior, dtype=np.float32)
        mn = np.full(n, prior, dtype=np.float32)
        if has.any():
            starts = indptr[:-1][has]
            mx[has] = np.maximum.reduceat(vals, starts).astype(np.float32)
            mn[has] = np.minimum.reduceat(vals, starts).astype(np.float32)
        mean = feats[:, 0]
        sq = _customer_feature(mat, enc**2, prior)
        feats[:, 6] = np.sqrt(np.maximum(sq - mean**2, 0))
        feats[:, 7] = mx
        feats[:, 8] = mn

        return feats
