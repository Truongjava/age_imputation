"""Buoc 5: dien age thieu va xuat file ket qua.

Output: outputs/customers_age_imputed.csv
    customer_id, age_original, age_filled, age_source
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src import config as C


def main() -> None:
    customers = pl.read_csv(
        C.CUSTOMERS_CSV,
        schema_overrides={"customer_id": pl.String, "age": pl.Float32},
    )

    missing_rows = np.load(C.OUTPUTS / "missing_rows.npy")
    pred = np.load(C.OUTPUTS / "pred_missing.npy")

    # cust_int = chi so dong goc cua customers.csv, nen thu tu dong duoc giu nguyen.
    age_original = customers.get_column("age").to_numpy()
    filled = age_original.astype(np.float32).copy()
    filled[missing_rows] = pred

    out = pl.DataFrame(
        {
            "customer_id": customers.get_column("customer_id"),
            "age_original": pl.Series(age_original).cast(pl.Float64).fill_nan(None),
            "age_filled": np.round(filled, 2),
            "age_source": np.where(np.isnan(age_original), "imputed", "observed"),
        }
    )

    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    out.write_csv(C.IMPUTED_CSV)
    print(f"wrote {out.height:,} rows -> {C.IMPUTED_CSV}")
    print(out.filter(pl.col("age_source") == "imputed").select("age_filled").describe())


if __name__ == "__main__":
    main()
