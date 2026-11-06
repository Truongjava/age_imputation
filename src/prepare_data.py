"""Buoc 1: chuan hoa du lieu goc thanh cac artifact nho, gon.

- customers_base.parquet : bang khach hang + chi so nguyen cust_int
- articles_encoded.parquet: ma hoa categorical cua articles (top-K + OTHER)
- transactions_compact.parquet: giao dich dang so nguyen (cust_int, art_int, day_idx)
"""

from __future__ import annotations

import json

import polars as pl

from src import config as C


def prepare_customers() -> pl.DataFrame:
    df = pl.read_csv(
        C.CUSTOMERS_CSV,
        schema_overrides={
            "customer_id": pl.String,
            "FN": pl.Float32,
            "Active": pl.Float32,
            "club_member_status": pl.String,
            "fashion_news_frequency": pl.String,
            "age": pl.Float32,
            "postal_code": pl.String,
        },
    )
    df = df.with_columns(
        pl.int_range(0, pl.len(), dtype=pl.Int32).alias("cust_int"),
        pl.col("Active").is_not_null().cast(pl.Int8).alias("active_flag"),
        pl.col("Active").fill_null(0).cast(pl.Int8),
        pl.col("age").clip(C.AGE_MIN, C.AGE_MAX),
    )
    return df


def prepare_articles() -> tuple[pl.DataFrame, dict[str, list[str]]]:
    df = pl.read_csv(
        C.ARTICLES_CSV,
        schema_overrides={"article_id": pl.String},
    ).select(["article_id"] + C.CAT_COLS)

    df = df.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("art_int"))

    maps: dict[str, list[str]] = {}
    enc_exprs = []
    for col in C.CAT_COLS:
        k = C.TOP_K[col]
        # Giu cac gia tri pho bien nhat, con lai gop thanh category "OTHER" (ma = k)
        top = (
            df.get_column(col)
            .fill_null("__NA__")
            .value_counts()
            .sort("count", descending=True)
            .head(k)
            .get_column(col)
            .to_list()
        )
        maps[col] = top + ["OTHER"]
        enc_exprs.append(
            pl.col(col)
            .fill_null("__NA__")
            .replace_strict(
                {v: i for i, v in enumerate(top)},
                default=k,
                return_dtype=pl.Int16,
            )
            .alias("c_" + col)
        )

    enc = df.select(["article_id", "art_int"] + enc_exprs)
    return enc, maps


def prepare_transactions(
    customers: pl.DataFrame, articles_enc: pl.DataFrame
) -> None:
    txn = pl.scan_csv(
        C.TXN_CSV,
        schema_overrides={
            "t_dat": pl.String,
            "customer_id": pl.String,
            "article_id": pl.String,
            "price": pl.Float32,
            "sales_channel_id": pl.Int8,
        },
    )

    compact = (
        txn.join(
            customers.select("customer_id", "cust_int").lazy(),
            on="customer_id",
            how="inner",
        )
        .join(
            articles_enc.select("article_id", "art_int").lazy(),
            on="article_id",
            how="inner",
        )
        .select(
            pl.col("cust_int").cast(pl.Int32),
            pl.col("art_int").cast(pl.Int32),
            (pl.col("t_dat").str.to_date("%Y-%m-%d") - pl.date(2018, 9, 20))
            .dt.total_days()
            .cast(pl.Int16)
            .alias("day_idx"),
            pl.col("price").cast(pl.Float32),
            pl.col("sales_channel_id").alias("channel"),
        )
    )

    compact.sink_parquet(
        C.TXN_PARQUET, compression="zstd", row_group_size=2_000_000, engine="streaming"
    )


def main() -> None:
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)

    customers = prepare_customers()
    customers.write_parquet(C.ARTIFACTS / "customers_base.parquet")
    print(f"customers: {customers.shape}")

    articles_enc, maps = prepare_articles()
    articles_enc.write_parquet(C.ARTIFACTS / "articles_encoded.parquet")
    C.CATEGORY_MAPS_JSON.write_text(json.dumps(maps, ensure_ascii=False, indent=2))
    print(f"articles:  {articles_enc.shape}")

    prepare_transactions(customers, articles_enc)
    n_txn = pl.scan_parquet(C.TXN_PARQUET).select(pl.len()).collect().item()
    print(f"transactions compact: {n_txn:,} rows -> {C.TXN_PARQUET}")


if __name__ == "__main__":
    main()
