"""Buoc 2: xay dung dac trung (feature) cho tung khach hang.

Xuat ra:
- artifacts/customer_base_features.parquet : thong ke giao dich + thuoc tinh khach hang
- artifacts/cat_counts_<col>.npz          : ma tran thua (sparse) dem luot mua theo category
- artifacts/feature_meta.json             : ten feature, so khach hang, so luong category
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy import sparse

from src import config as C

MAX_DAY = 733  # 2020-09-22 - 2018-09-20


def base_features(customers: pl.DataFrame) -> pl.DataFrame:
    txn = pl.scan_parquet(C.TXN_PARQUET)

    agg = (
        txn.group_by("cust_int")
        .agg(
            pl.len().alias("n_txn"),
            pl.col("art_int").n_unique().alias("n_uniq_art"),
            pl.col("price").sum().alias("total_spend"),
            pl.col("price").mean().alias("mean_price"),
            pl.col("price").median().alias("median_price"),
            pl.col("price").max().alias("max_price"),
            pl.col("price").std().alias("std_price"),
            pl.col("day_idx").min().alias("first_day"),
            pl.col("day_idx").max().alias("last_day"),
            pl.col("day_idx").n_unique().alias("n_days"),
            (pl.col("channel") == 2).mean().alias("online_share"),
            ((pl.col("day_idx") + 3) % 7).is_in([5, 6]).mean().alias("weekend_share"),
            (pl.col("day_idx") >= MAX_DAY - 365).sum().alias("n_txn_365d"),
            pl.col("price").filter(pl.col("day_idx") >= MAX_DAY - 365).mean().alias("mean_price_365d"),
            (pl.col("day_idx") >= MAX_DAY - 90).sum().alias("n_txn_90d"),
        )
        .collect(engine="streaming")
    )

    df = customers.select(
        "cust_int",
        "age",
        "club_member_status",
        "fashion_news_frequency",
        "active_flag",
        "Active",
        "postal_code",
    ).join(agg, on="cust_int", how="left")

    postal_size = df.group_by("postal_code").agg(pl.len().alias("postal_size"))
    df = df.join(postal_size, on="postal_code", how="left").with_columns(
        pl.col("n_txn").fill_null(0),
        (MAX_DAY - pl.col("last_day")).alias("recency"),
        (pl.col("last_day") - pl.col("first_day")).alias("tenure"),
        (pl.col("n_txn") / pl.col("n_days").clip(1, None)).alias("txn_per_active_day"),
        (1 - pl.col("n_uniq_art") / pl.col("n_txn").clip(1, None)).alias("repeat_item_ratio"),
        (pl.col("total_spend") / pl.col("n_txn").clip(1, None)).alias("spend_per_txn"),
        (pl.col("n_txn_365d") / pl.col("n_txn").clip(1, None)).alias("recent_activity_ratio"),
        (pl.col("mean_price_365d") / pl.col("mean_price")).alias("price_trend"),
        pl.col("total_spend").log1p().alias("log_total_spend"),
        pl.col("postal_size").log1p().alias("log_postal_size"),
        pl.col("club_member_status").fill_null("__NA__").cast(pl.Categorical).to_physical().cast(pl.Int8).alias("club_code"),
        pl.col("fashion_news_frequency").fill_null("__NA__").cast(pl.Categorical).to_physical().cast(pl.Int8).alias("news_code"),
    ).drop("last_day", "first_day", "postal_code", "age")

    num_cols = [c for c, dt in df.schema.items() if dt.is_numeric() and c != "cust_int"]
    return df.select(["cust_int"] + num_cols)


def cust_article_matrix() -> None:
    """Ma tran thua cust x article (so luot mua cua moi cap) - dung cho target encoding."""
    n_cust = pl.scan_parquet(C.TXN_PARQUET).select(pl.col("cust_int").max()).collect().item() + 1
    n_art = pl.scan_parquet(C.TXN_PARQUET).select(pl.col("art_int").max()).collect().item() + 1
    pairs = (
        pl.scan_parquet(C.TXN_PARQUET)
        .group_by(["cust_int", "art_int"])
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    mat = sparse.csr_matrix(
        (pairs.get_column("n").to_numpy().astype(np.float32),
         (pairs.get_column("cust_int").to_numpy(), pairs.get_column("art_int").to_numpy())),
        shape=(n_cust, n_art),
        dtype=np.float32,
    )
    sparse.save_npz(C.ARTIFACTS / "cust_article.npz", mat, compressed=True)
    print(f"  cust x article: nnz={mat.nnz:,}")


def level_counts(levels: pl.DataFrame) -> dict[str, int]:
    """Ma tran dem theo cac muc phan loai day du (khong gop OTHER)."""
    n_cust = pl.scan_parquet(C.TXN_PARQUET).select(pl.col("cust_int").max()).collect().item() + 1
    sizes: dict[str, int] = {}
    level_cols = [c for c in levels.columns if c != "art_int"]
    for col in level_cols:
        n_lvl = int(levels.get_column(col).max()) + 1
        sub = levels.select("art_int", col).lazy()
        counts = (
            pl.scan_parquet(C.TXN_PARQUET)
            .join(sub, on="art_int", how="left")
            .group_by(["cust_int", col])
            .agg(pl.len().alias("n"))
            .collect(engine="streaming")
        )
        rows = counts.get_column("cust_int").to_numpy().astype(np.int32)
        cols = counts.get_column(col).to_numpy().astype(np.int32)
        vals = counts.get_column("n").to_numpy().astype(np.float32)
        mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cust, n_lvl), dtype=np.float32)
        mat.sum_duplicates()
        sparse.save_npz(C.ARTIFACTS / f"lvl_{col}.npz", mat, compressed=True)
        sizes[col] = n_lvl
        print(f"  {col}: {n_lvl} levels, nnz={mat.nnz:,}")
    return sizes


def category_counts(articles_enc: pl.DataFrame) -> dict[str, tuple[sparse.csr_matrix, list[str]]]:
    n_cust = pl.scan_parquet(C.TXN_PARQUET).select(pl.col("cust_int").max()).collect().item() + 1
    out: dict[str, tuple[sparse.csr_matrix, list[str]]] = {}
    cat_df = articles_enc.select(["art_int"] + ["c_" + c for c in C.CAT_COLS]).lazy()

    for col in C.CAT_COLS:
        code_col = "c_" + col
        n_cat = C.TOP_K[col] + 1
        counts = (
            pl.scan_parquet(C.TXN_PARQUET)
            .join(cat_df, on="art_int", how="left")
            .group_by(["cust_int", code_col])
            .agg(pl.len().alias("n"))
            .collect(engine="streaming")
        )
        rows = counts.get_column("cust_int").to_numpy().astype(np.int32)
        cols = counts.get_column(code_col).to_numpy().astype(np.int32)
        vals = counts.get_column("n").to_numpy().astype(np.float32)
        mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cust, n_cat), dtype=np.float32)
        mat.sum_duplicates()
        sparse.save_npz(C.ARTIFACTS / f"cat_counts_{col}.npz", mat, compressed=True)
        out[col] = (mat, [f"{col}__{i}" for i in range(n_cat)])
        print(f"  {col}: {n_cat} categories, nnz={mat.nnz:,}")
    return out


def main() -> None:
    customers = pl.read_parquet(C.ARTIFACTS / "customers_base.parquet")
    articles_enc = pl.read_parquet(C.ARTIFACTS / "articles_encoded.parquet")

    print("base features...")
    base = base_features(customers)
    base.write_parquet(C.ARTIFACTS / "customer_base_features.parquet")
    print(f"  {base.shape} -> {list(base.columns)}")

    print("category counts...")
    cats = category_counts(articles_enc)

    print("cust x article matrix...")
    cust_article_matrix()

    print("full-cardinality level counts...")
    levels = pl.read_parquet(C.ARTIFACTS / "articles_levels.parquet")
    level_sizes = level_counts(levels)

    meta = {
        "n_customers": customers.height,
        "max_day": MAX_DAY,
        "cat_cols": C.CAT_COLS,
        "cat_feature_names": {c: names for c, (_, names) in cats.items()},
        "level_cols": list(level_sizes),
        "level_sizes": level_sizes,
        "base_features": [c for c in base.columns if c != "cust_int"],
        "n_cat_features": sum(len(names) for _, names in cats.values()),
    }
    C.FEATURE_META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"total cat features: {meta['n_cat_features']}")


if __name__ == "__main__":
    main()
