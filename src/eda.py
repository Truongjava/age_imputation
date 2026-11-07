"""Buoc 0: EDA - phan tich du lieu truoc khi chon phuong phap.

Tra loi cac cau hoi:
  1. Thieu age bao nhieu, co phai thieu ngau nhien (MCAR) khong?
  2. Phan phoi age: hinh dang, chot, bien.
  3. Dac trung nao thuc su tuong quan voi age?
  4. Tin hieu article-level manh hay yeu? postal_code co tin hieu khong?
  5. Da cong tuyen / trung lap giua cac dac trung?
  6. Tu do chon: muc tieu loss, mo hinh, feature set, cach danh gia.

Xuat: reports/eda/*.png, reports/eda_report.md, artifacts/eda_stats.json
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import sparse
from scipy.stats import spearmanr

from src import config as C

EDA_DIR = C.EDA_DIR
STATS_JSON = C.ARTIFACTS / "eda_stats.json"

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load() -> tuple[np.ndarray, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    y = np.load(C.ARTIFACTS / "y_age.npy")
    base = pl.read_parquet(C.ARTIFACTS / "customer_base_features.parquet").sort("cust_int")
    cust = pl.read_parquet(C.ARTIFACTS / "customers_base.parquet").sort("cust_int")
    arts = pl.read_parquet(C.ARTIFACTS / "articles_levels.parquet").sort("art_int")
    return y, base, cust, arts


# ---------------------------------------------------------------- 1. missingness
def missingness(y: np.ndarray, cust: pl.DataFrame, base: pl.DataFrame) -> dict:
    miss = np.isnan(y)
    df = cust.select("club_member_status", "fashion_news_frequency", "active_flag", "Active").with_columns(
        pl.Series("missing", miss)
    )

    by_club = (
        df.group_by("club_member_status")
        .agg(pl.len().alias("n"), pl.col("missing").mean().alias("miss_rate"))
        .sort("n", descending=True)
    )
    by_news = (
        df.with_columns(pl.col("fashion_news_frequency").str.to_uppercase())
        .group_by("fashion_news_frequency")
        .agg(pl.len().alias("n"), pl.col("missing").mean().alias("miss_rate"))
        .sort("n", descending=True)
    )
    by_active = (
        df.group_by("active_flag")
        .agg(pl.len().alias("n"), pl.col("missing").mean().alias("miss_rate"))
        .sort("active_flag")
    )

    n_txn = base.get_column("n_txn").to_numpy()
    activity = {
        "labeled_median_txn": float(np.median(n_txn[~miss])),
        "missing_median_txn": float(np.median(n_txn[miss])),
        "labeled_mean_txn": float(n_txn[~miss].mean()),
        "missing_mean_txn": float(n_txn[miss].mean()),
        "labeled_zero_txn_rate": float((n_txn[~miss] == 0).mean()),
        "missing_zero_txn_rate": float((n_txn[miss] == 0).mean()),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, table, col, title in [
        (axes[0], by_club, "club_member_status", "Missing rate by club member status"),
        (axes[1], by_news, "fashion_news_frequency", "Missing rate by news frequency"),
        (axes[2], by_active, "active_flag", "Missing rate by active flag"),
    ]:
        labels = [
            ("no Active" if v == 0 else "Active") if col == "active_flag" else str(v)
            for v in table.get_column(col).to_list()
        ]
        rates = (table.get_column("miss_rate").to_numpy() * 100).round(2)
        bars = ax.bar(labels, rates, color="#4C78A8")
        ax.bar_label(bars, fmt="%.2f%%", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("missing age (%)")
        ax.set_ylim(0, max(rates.max() * 1.3, 1.2))
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
    fig.suptitle("Age missingness is spread across all segments (overall 1.16%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "fig1_missingness.png")
    plt.close(fig)

    return {
        "overall_missing_rate": float(miss.mean()),
        "by_club_member_status": by_club.to_dicts(),
        "by_fashion_news_frequency": by_news.to_dicts(),
        "by_active_flag": by_active.to_dicts(),
        "activity": activity,
    }


# ---------------------------------------------------------------- 2. target
def target_distribution(y: np.ndarray) -> dict:
    lab = y[~np.isnan(y)]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(lab, bins=np.arange(15.5, 100.5, 1), color="#4C78A8")
    ax.axvline(np.median(lab), color="#E45756", ls="--", label=f"median = {np.median(lab):.0f}")
    ax.set_xlabel("age")
    ax.set_ylabel("customers")
    ax.set_title("Age distribution (labeled customers): right-skewed, bounded at 16")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "fig2_age_distribution.png")
    plt.close(fig)

    q = np.percentile(lab, [1, 5, 25, 50, 75, 95, 99])
    vals, cnts = np.unique(lab, return_counts=True)
    top_vals = vals[np.argsort(-cnts)][:6]
    return {
        "count": int(lab.size),
        "mean": float(lab.mean()),
        "std": float(lab.std()),
        "skew": float(((lab - lab.mean()) ** 3).mean() / lab.std() ** 3),
        "quantiles": {f"p{p}": float(v) for p, v in zip([1, 5, 25, 50, 75, 95, 99], q)},
        "at_min_age": int((lab == C.AGE_MIN).sum()),
        "at_99": int((lab == 99).sum()),
        "most_common_ages": {int(v): int(c) for v, c in zip(top_vals, cnts[np.argsort(-cnts)][:6])},
        "share_20_39": float(((lab >= 20) & (lab < 40)).mean()),
        "share_50_plus": float((lab >= 50).mean()),
    }


# ---------------------------------------------------------------- 3. signals
def numeric_signals(y: np.ndarray, base: pl.DataFrame) -> dict:
    num_cols = [
        c
        for c, dt in base.schema.items()
        if dt.is_numeric() and c not in {"cust_int", "Active", "club_code", "news_code"}
    ]
    lab = ~np.isnan(y)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(lab)
    if idx.size > 300_000:
        idx = rng.choice(idx, 300_000, replace=False)
    yy = y[idx]

    rows = []
    for col in num_cols:
        x = base.get_column(col).to_numpy().astype(np.float64)[idx]
        good = np.isfinite(x)
        rho = spearmanr(x[good], yy[good]).statistic
        rows.append((col, float(rho), abs(float(rho))))
    rows.sort(key=lambda t: -t[2])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = rows[:15][::-1]
    ax.barh([t[0] for t in top], [t[1] for t in top], color="#4C78A8")
    ax.set_xlabel("Spearman correlation with age")
    ax.set_title("Behavioral features correlate weakly with age (|rho| < 0.2)")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "fig3_correlations.png")
    plt.close(fig)

    # Mean age theo nhom thuoc tinh "ai dang mua" (index group chiem uu the)
    ig = sparse.load_npz(C.ARTIFACTS / "cat_counts_index_group_name.npz").tocsr()
    names = json.loads(C.CATEGORY_MAPS_JSON.read_text())["index_group_name"]
    totals = np.asarray(ig.sum(axis=1)).ravel()
    dominant = np.asarray(ig.argmax(axis=1)).ravel()
    has_txn = totals > 0
    rows_group = []
    for gi, name in enumerate(names):
        sel = (dominant == gi) & has_txn & lab
        if sel.sum() < 100:
            continue
        rows_group.append(
            {
                "index_group": name,
                "n": int(sel.sum()),
                "share": float(sel.sum() / (has_txn & lab).sum()),
                "mean_age": float(y[sel].mean()),
                "median_age": float(np.median(y[sel])),
            }
        )
    rows_group.sort(key=lambda r: -r["n"])

    # Mean age theo decile cua mean_price va n_txn
    price = base.get_column("mean_price").to_numpy()
    ntxn = base.get_column("n_txn").to_numpy()

    def decile_profile(vals: np.ndarray, name: str) -> list[dict]:
        v = np.where(np.isfinite(vals), vals, np.nan)
        sel = lab & np.isfinite(v) & (v > 0)
        edges = np.quantile(v[sel], np.linspace(0, 1, 11))
        edges[0], edges[-1] = -np.inf, np.inf
        bin_id = np.digitize(v, edges[1:-1])
        out = []
        for b in range(10):
            m = sel & (bin_id == b)
            out.append({"decile": b + 1, "n": int(m.sum()), "mean_age": float(y[m].mean())})
        return out

    price_prof = decile_profile(price, "mean_price")
    txn_prof = decile_profile(ntxn, "n_txn")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot([d["decile"] for d in price_prof], [d["mean_age"] for d in price_prof], "o-")
    axes[0].set_title("Mean age by average item price decile")
    axes[0].set_xlabel("price decile")
    axes[0].set_ylabel("mean age")
    axes[1].plot([d["decile"] for d in txn_prof], [d["mean_age"] for d in txn_prof], "o-", color="#E45756")
    axes[1].set_title("Mean age by transaction-count decile")
    axes[1].set_xlabel("activity decile")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "fig4_price_activity.png")
    plt.close(fig)

    return {
        "spearman_top": [{"feature": r[0], "rho": round(r[1], 3)} for r in rows[:20]],
        "max_abs_rho": rows[0][2],
        "n_features_abs_rho_gt_0.1": int(sum(1 for r in rows if r[2] > 0.1)),
        "age_by_index_group": rows_group,
        "price_deciles": price_prof,
        "txn_deciles": txn_prof,
    }


# ------------------------------------------------- 4. article vs postal signal
def article_and_postal_signal(y: np.ndarray, arts: pl.DataFrame, cust: pl.DataFrame) -> dict:
    mat = sparse.load_npz(C.ARTIFACTS / "cust_article.npz").tocsr()
    prior = float(np.nanmedian(y))
    lab = ~np.isnan(y)

    # Article mean buyer age (fit tren toan bo nhan) de mo ta, khong dung de danh gia.
    sums = np.asarray(mat[lab].T @ y[lab]).ravel()
    cnts = np.asarray(mat[lab].sum(axis=0)).ravel()
    alpha = 10.0
    art_mean = (sums + prior * alpha) / (cnts + alpha)
    ok = cnts >= 50
    ig_names = json.loads(C.CATEGORY_MAPS_JSON.read_text())["index_group_name"]
    # articles_levels.parquet dùng mã categorical theo thứ tự xuất hiện (không theo thứ tự
    # trong category_maps.json) -> ánh xạ ngược mã -> tên trước khi vẽ.
    lvl = pl.read_parquet(C.ARTIFACTS / "articles_levels.parquet").select(
        "art_int", "c_index_group_name"
    )
    joined = (
        pl.read_csv(C.ARTICLES_CSV, schema_overrides={"article_id": pl.String})
        .select("article_id", "index_group_name")
        .join(
            pl.read_parquet(C.ARTIFACTS / "articles_encoded.parquet").select(
                "article_id", "art_int"
            ),
            on="article_id",
            how="inner",
        )
        .join(lvl, on="art_int", how="left")
        .sort("art_int")
    )
    code_to_name = {
        int(code): name
        for code, name in joined.group_by("c_index_group_name")
        .agg(pl.col("index_group_name").first())
        .rows()
    }
    art_ig = joined.get_column("c_index_group_name").to_numpy()[: cnts.size]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].hist(art_mean[ok], bins=40, color="#72B7B2")
    axes[0].set_title(f"Article mean buyer age (articles with >=50 buyers, n={int(ok.sum()):,})")
    axes[0].set_xlabel("mean age")
    axes[0].set_ylabel("articles")
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    for gi, (code, name) in enumerate(sorted(code_to_name.items())):
        sel = ok & (art_ig == code)
        if sel.sum() > 10:
            axes[1].scatter(
                cnts[sel], art_mean[sel], s=4, alpha=0.45, label=f"{name} (n={int(sel.sum()):,})",
                color=colors[gi % len(colors)],
            )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("buyers per article (log)")
    axes[1].set_ylabel("mean buyer age")
    axes[1].set_title("Kids/women/men articles separate clearly by age")
    axes[1].legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "fig5_article_signal.png")
    plt.close(fig)

    # Danh gia OOF nhanh: article TE va postal TE tren 1 split 700k/rest
    rng = np.random.default_rng(42)
    labeled = np.flatnonzero(lab)
    perm = rng.permutation(labeled)
    tr, ev = perm[:700_000], perm[700_000:]

    def cv_mae(feat_matrix: sparse.csr_matrix, alpha_: float) -> float:
        s = np.asarray(feat_matrix[tr].T @ y[tr]).ravel()
        c = np.asarray(feat_matrix[tr].sum(axis=0)).ravel()
        enc = (s + prior * alpha_) / (c + alpha_)
        num = np.asarray(feat_matrix[ev] @ enc).ravel()
        den = np.asarray(feat_matrix[ev].sum(axis=1)).ravel()
        pred = np.where(den > 0, num / np.where(den > 0, den, 1), prior)
        return float(np.abs(pred - y[ev]).mean())

    postal_code = (
        cust.select("postal_code")
        .get_column("postal_code")
        .cast(pl.Categorical)
        .to_physical()
        .to_numpy()
        .astype(np.int64)
    )
    n_postal = int(postal_code.max()) + 1
    s = np.bincount(postal_code[tr], weights=y[tr], minlength=n_postal)
    c = np.bincount(postal_code[tr], minlength=n_postal)
    postal_enc = (s + prior * 50.0) / (c + 50.0)
    postal_mae = float(np.abs(postal_enc[postal_code[ev]] - y[ev]).mean())

    results = {
        "global_median_mae": float(np.abs(prior - y[ev]).mean()),
        "postal_code_te_mae": postal_mae,
        "article_te_mae": cv_mae(mat, 10.0),
        "postal_coverage": float((c[postal_code[ev]] > 0).mean()),
        "postal_mean_customers": float(np.mean(c[c > 0])),
        "article_age_std": float(np.std(art_mean[ok])),
        "article_age_range": [float(np.percentile(art_mean[ok], 1)), float(np.percentile(art_mean[ok], 99))],
        "corr_customer_article_te": None,
    }

    # Tuong quan giua "article TE trung binh cua khach" va age thuc
    te_all = (sums + prior * alpha) / (cnts + alpha)
    num = np.asarray(mat[ev] @ te_all).ravel()
    den = np.asarray(mat[ev].sum(axis=1)).ravel()
    cust_feat = np.where(den > 0, num / np.where(den > 0, den, 1), prior)
    good = den > 0
    results["corr_customer_article_te"] = float(np.corrcoef(cust_feat[good], y[ev][good])[0, 1])

    fig, ax = plt.subplots(figsize=(7, 3))
    labels = ["global median", "postal-code TE", "article TE"]
    vals = [results["global_median_mae"], postal_mae, results["article_te_mae"]]
    bars = ax.barh(labels, vals, color=["#B0B0B0", "#E45756", "#54A24B"])
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xlabel("MAE (years), 700k/656k hold-out split")
    ax.set_title("Only purchasing behavior carries signal; postal code does not")
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "fig6_signal_mae.png")
    plt.close(fig)
    return results


# ---------------------------------------------------------- 5. redundancy / quality
def redundancy_and_quality(y: np.ndarray, base: pl.DataFrame) -> dict:
    pairs = [
        ("total_spend", "log_total_spend"),
        ("n_txn", "total_spend"),
        ("n_txn", "n_txn_365d"),
        ("Active", "active_flag"),
        ("mean_price", "median_price"),
        ("postal_size", "log_postal_size"),
    ]
    out = []
    for a, b in pairs:
        x = base.get_column(a).to_numpy().astype(float)
        z = base.get_column(b).to_numpy().astype(float)
        good = np.isfinite(x) & np.isfinite(z)
        out.append(
            {
                "pair": f"{a} ~ {b}",
                "spearman": round(float(spearmanr(x[good], z[good]).statistic), 4),
                "identical": bool(np.allclose(x[good], z[good], equal_nan=True)),
            }
        )
    lab = y[~np.isnan(y)]
    return {
        "redundant_pairs": out,
        "age_at_16_rate": float((lab == 16).mean()),
        "age_at_99_rate": float((lab == 99).mean()),
        "age_integer_rate": float(np.mean(lab == np.round(lab))),
    }


def temporal_and_quality_customers(customers: pl.DataFrame) -> dict:
    """Kiem tra chat luong du lieu: referential integrity, duplicates, whales, xu huong thoi gian, mau sac."""
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
    lab = customers.select("customer_id", "age").drop_nulls("age")
    articles = pl.read_csv(C.ARTICLES_CSV, schema_overrides={"article_id": pl.String})

    # A) referential integrity
    tx_keys = txn.select("customer_id", "article_id").unique().collect(engine="streaming")
    cust_set = customers["customer_id"]
    art_set = articles["article_id"]
    n_bad_cust = int((~tx_keys["customer_id"].is_in(cust_set)).sum())
    n_bad_art = int((~tx_keys["article_id"].is_in(art_set)).sum())

    # B) duplicates
    dup = (
        txn.group_by(["customer_id", "article_id", "t_dat", "price", "sales_channel_id"])
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    n_dup_rows = int((dup["n"] - 1).sum())

    # C) whales
    per_cust = txn.group_by("customer_id").agg(n=pl.len()).collect(engine="streaming")
    pcs = np.sort(per_cust["n"].to_numpy())

    # D) temporal trend
    monthly = (
        txn.with_columns(month=pl.col("t_dat").str.slice(0, 7))
        .join(lab.lazy(), on="customer_id", how="inner")
        .group_by("month")
        .agg(n=pl.len(), mean_age=pl.col("age").mean())
        .sort("month")
        .collect(engine="streaming")
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].bar(range(monthly.height), monthly["n"].to_numpy(), color="#4C78A8")
    axes[0].set_ylabel("giao dịch/tháng")
    axes[0].set_title("Giao dịch theo tháng — tăng vọt 2020-04→07 (sau giãn cách COVID)")
    axes[1].plot(range(monthly.height), monthly["mean_age"].to_numpy(), "o-", color="#E45756", ms=3)
    axes[1].set_ylabel("mean age")
    axes[1].set_xlabel("tháng")
    axes[1].set_xticks(range(0, monthly.height, 3))
    axes[1].set_xticklabels(monthly["month"].to_list()[::3], rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    fig.savefig(EDA_DIR / "fig8_temporal.png")
    plt.close(fig)

    # E) colour signal
    colour = (
        txn.join(articles.select("article_id", "colour_group_name").lazy(), on="article_id", how="left")
        .join(lab.lazy(), on="customer_id", how="inner")
        .group_by("colour_group_name")
        .agg(n=pl.len(), mean_age=pl.col("age").mean())
        .filter(pl.col("n") > 100_000)
        .sort("mean_age")
        .collect(engine="streaming")
    )

    return {
        "referential_integrity": {"bad_customer_keys": n_bad_cust, "bad_article_keys": n_bad_art},
        "duplicates": {
            "n_duplicate_rows": n_dup_rows,
            "share_pct": round(n_dup_rows / 31_788_324 * 100, 2),
            "max_repeat": int(dup["n"].max()),
        },
        "activity_tails": {
            "p90": float(np.percentile(pcs, 90)),
            "p99": float(np.percentile(pcs, 99)),
            "p99.9": float(np.percentile(pcs, 99.9)),
            "max": int(pcs.max()),
            "whales_ge_1000": int((pcs >= 1000).sum()),
        },
        "temporal": {
            "mean_age_first12": float(monthly["mean_age"].head(12).mean()),
            "mean_age_last12": float(monthly["mean_age"].tail(12).mean()),
            "peak_month_n": int(monthly["n"].max()),
            "monthly": monthly.to_dicts(),
        },
        "colour_signal": {
            "spread_years": float(colour["mean_age"].max() - colour["mean_age"].min()),
            "n_colours": colour.height,
            "by_colour": colour.to_dicts(),
        },
    }


def missingness_impact(y: np.ndarray, base: pl.DataFrame) -> dict:
    """Thieu age co tuong quan voi muc do hoat dong -> anh huong gi khi dien?"""
    n_txn = base.get_column("n_txn").to_numpy()
    miss = np.isnan(y)
    labeled = ~miss

    edges = np.concatenate([[-1], np.percentile(n_txn[labeled], [20, 40, 60, 80, 95]), [np.inf]])
    bin_id = np.digitize(n_txn, edges[1:-1])
    rows = []
    oof_path = C.OUTPUTS / "oof_pred.npy"
    oof = np.load(oof_path) if oof_path.exists() else None
    for b in range(6):
        sel = (bin_id == b) & labeled
        row = {
            "activity_bin": b + 1,
            "txn_range": [float(edges[b]), float(edges[b + 1]) if np.isfinite(edges[b + 1]) else None],
            "n_labeled": int(sel.sum()),
            "share_missing": float(miss[bin_id == b].mean()),
            "mean_age": float(y[sel].mean()),
        }
        if oof is not None:
            p = np.clip(oof[sel], C.AGE_MIN, C.AGE_MAX)
            row["mae"] = float(np.abs(p - y[sel]).mean())
        rows.append(row)

    # Uoc luong MAE tren nhom thieu age: tron theo cau truc hoat dong cua chinh nhom do
    miss_share = np.array([float((bin_id == b)[miss].mean()) for b in range(6)])
    mae_by_bin = np.array([r.get("mae", np.nan) for r in rows])
    expected_mae_missing = float(np.nansum(miss_share * mae_by_bin))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["share_missing"] * 100 for r in rows], width=0.4, label="missing age (%)", color="#E45756")
    ax2 = ax.twinx()
    ax2.plot(xs + 0.2, [r.get("mae", np.nan) for r in rows], "o-", color="#54A24B", label="OOF MAE")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['txn_range'][0]:.0f}-{r['txn_range'][1]:.0f}" if r["txn_range"][1] else f">{r['txn_range'][0]:.0f}" for r in rows], fontsize=7)
    ax.set_xlabel("transactions per customer")
    ax.set_ylabel("missing age rate (%)")
    ax2.set_ylabel("OOF MAE (years)")
    ax.set_title("Missing-age customers are less active -> higher expected error")
    ax.grid(False)
    ax2.grid(False)
    fig.tight_layout()
    fig.savefig(EDA_DIR / "fig7_missingness_impact.png")
    plt.close(fig)
    return {
        "by_activity_bin": rows,
        "missing_composition_by_bin": [float(v) for v in miss_share],
        "expected_mae_on_missing": expected_mae_missing,
        "median_txn_labeled": float(np.median(n_txn[labeled])),
        "median_txn_missing": float(np.median(n_txn[miss])),
    }


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    y, base, cust, arts = load()

    print("1/5 missingness...")
    miss_stats = missingness(y, cust, base)
    print("2/5 target distribution...")
    target_stats = target_distribution(y)
    print("3/5 numeric signals...")
    signal_stats = numeric_signals(y, base)
    print("4/5 article vs postal signal...")
    art_stats = article_and_postal_signal(y, arts, cust)
    print("5/5 redundancy...")
    qual_stats = redundancy_and_quality(y, base)

    print("+ missingness impact...")
    impact_stats = missingness_impact(y, base)

    print("+ temporal & quality (duplicates, whales, colour)...")
    temporal_stats = temporal_and_quality_customers(cust)

    stats = {
        "missingness": miss_stats,
        "target": target_stats,
        "signals": signal_stats,
        "article_vs_postal": art_stats,
        "quality": qual_stats,
        "missingness_impact": impact_stats,
        "temporal_quality": temporal_stats,
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {STATS_JSON}")


if __name__ == "__main__":
    main()
