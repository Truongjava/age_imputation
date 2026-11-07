"""Buoc 6: sinh bao cao Markdown tu metrics.json."""

from __future__ import annotations

import json

import numpy as np

from src import config as C

TE_NAMES = [
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


def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def pct(v: float, nd: int = 1) -> str:
    return f"{v * 100:.{nd}f}%"


def age_group_table() -> str:
    """Sai số out-of-fold theo từng nhóm tuổi thật."""
    y = np.load(C.ARTIFACTS / "y_age.npy")
    oof = np.load(C.OUTPUTS / "oof_pred.npy")
    mask = ~np.isnan(y) & ~np.isnan(oof)
    yt, yp = y[mask], np.clip(oof[mask], C.AGE_MIN, C.AGE_MAX)
    groups = np.digitize(yt, C.AGE_BINS[1:-1])
    lines = []
    for gi, label in enumerate(C.AGE_BIN_LABELS):
        sel = groups == gi
        if not sel.any():
            continue
        mae = np.abs(yt[sel] - yp[sel]).mean()
        bias = (yp[sel] - yt[sel]).mean()
        lines.append(
            f"| {label} | {int(sel.sum()):,} | {sel.mean() * 100:.1f}% | {mae:.2f} | {bias:+.2f} |"
        )
    return "\n".join(lines)


def eda_summary() -> str:
    """Tóm tắt EDA (nếu đã chạy src/eda.py) để đưa vào báo cáo chính."""
    path = C.ARTIFACTS / "eda_stats.json"
    if not path.exists():
        return ""
    s = json.loads(path.read_text())
    sig = s["signals"]
    art = s["article_vs_postal"]
    imp = s["missingness_impact"]
    miss = s["missingness"]
    tgt = s["target"]
    club_null = next(
        (r for r in miss["by_club_member_status"] if r["club_member_status"] is None), None
    )
    club_active = next(
        (r for r in miss["by_club_member_status"] if r["club_member_status"] == "ACTIVE"), None
    )
    return f"""## 3. EDA dẫn dắt thiết kế (tóm tắt)

Báo cáo EDA đầy đủ: [`reports/eda_report.md`](eda_report.md) — biểu đồ trong `reports/eda/`.

| Phát hiện từ EDA | Số liệu | Quyết định kéo theo |
|---|---|---|
| Thiếu age **không ngẫu nhiên** | {club_null['miss_rate'] * 100:.1f}% ở nhóm `club_member_status` null so với {club_active['miss_rate'] * 100:.2f}% ở ACTIVE; median giao dịch {imp['median_txn_missing']:.0f} so với {imp['median_txn_labeled']:.0f} | Báo cáo thêm MAE kỳ vọng trên nhóm thiếu (~{imp['expected_mae_on_missing']:.1f}) |
| Target lệch phải, đuôi dài | skew {tgt['skew']:.2f}, p99 = {tgt['quantiles']['p99']:.0f}, min = {C.AGE_MIN} | Hồi quy + loss L1 + clip [{C.AGE_MIN}, {C.AGE_MAX}] |
| Đặc trưng RFM đơn biến rất yếu | max \\|ρ\\| = {sig['max_abs_rho']:.3f} | Không dùng hồi quy tuyến tính; dùng GBDT + đặc trưng sở thích sản phẩm |
| Tín hiệu thật nằm ở **article** | TE article: MAE {art['article_te_mae']:.2f} (median {art['global_median_mae']:.2f}), corr {art['corr_customer_article_te']:.2f} | Ma trận cust × article + TE nhiều mức theo fold |
| `postal_code` không có tín hiệu | MAE {art['postal_code_te_mae']:.2f} ≈ median {art['global_median_mae']:.2f} | Loại `postal_code` khỏi feature |
| Sai số phụ thuộc mức hoạt động | MAE 6.0 (>91 giao dịch) → 11.2 (≤3 giao dịch) | Không quảng bá MAE 8.8 cho toàn bộ nhóm thiếu |

"""


def main() -> None:
    m = json.loads(C.METRICS_JSON.read_text())
    cv = m["cv"]
    base = m["baselines"]
    n_total = m["n_customers"]
    n_lab, n_mis = m["n_labeled"], m["n_missing"]

    rows = "\n".join(
        f"| {r['fold']} | {fmt(r['mae'])} | {fmt(r['rmse'])} | {fmt(r['r2'], 4)} "
        f"| {pct(r['bucket_acc'])} | {pct(r['within_5y'])} | {pct(r['within_10y'])} | {r['best_iter']} |"
        for r in m["folds"]
    )
    imp_rows = "\n".join(
        f"| {i + 1} | `{d['feature']}` | {d['gain']:,.0f} |"
        for i, d in enumerate(m["top_importance"])
    )
    obs, imp = m["observed_stats"], m["imputed_stats"]
    p = m["lgb_params"]
    base_names = [f for f in m["features"] if f not in TE_NAMES]
    n_share = sum(1 for f in base_names if "__" in f)
    n_behaviour = len(base_names) - n_share
    run_min = m["runtime_sec"] / 60
    best_iter_mean = sum(m["best_iterations"]) / len(m["best_iterations"])

    report = f"""# Age Imputation — H&M Personalized Fashion Recommendations

## 1. Bài toán

| | |
|---|---|
| Khách hàng | {n_total:,} |
| Có `age` | {n_lab:,} ({n_lab / n_total * 100:.2f}%) |
| **Thiếu `age`** | **{n_mis:,} ({n_mis / n_total * 100:.2f}%)** |
| Giao dịch dùng làm feature | 31,788,324 (2018-09-20 → 2020-09-22) |
| Khoảng tuổi | {C.AGE_MIN}–{C.AGE_MAX} |

Mục tiêu: dự đoán `age` cho {n_mis:,} khách thiếu, dựa trên hành vi mua sắm và thuộc tính tài khoản.

## 2. Kiến trúc pipeline

```mermaid
flowchart TD
    A[customers.csv / articles.csv / transactions_train.csv] --> Z[eda.py<br/>EDA: missingness, phân phối, tín hiệu, trùng lặp]
    A --> B[prepare_data.py<br/>chuẩn hoá, mã hoá về số nguyên]
    Z -.->|quyết định feature & loss| B
    B --> C[transactions_compact.parquet<br/>31.8M dòng]
    B --> D[customers_base.parquet<br/>articles_encoded.parquet]
    C --> E[features.py<br/>RFM + tỷ trọng category + ma trận cust x article]
    D --> E
    E --> F[base features + cat_counts / lvl_* / cust_article]
    F --> G[dataset.py<br/>X.npy float32 {n_total / 1e6:.2f}M x 96]
    F --> H[target_encoding.py<br/>encoding theo fold, chống rò rỉ]
    G --> I[train.py<br/>5-fold CV, LightGBM L1]
    H --> I
    I --> J[impute.py -> customers_age_imputed.csv]
    I --> K[report.py -> báo cáo này]
```

Mỗi bước ghi artifact ra đĩa (Parquet/NPZ/NPY) nên chạy lại không phải đọc lại CSV gốc 3.5 GB.

### Đặc trưng ({m['n_features']} chiều)

**1) Hành vi giao dịch (RFM + xu hướng)** — {n_behaviour} chiều:
`n_txn`, `n_uniq_art`, `total_spend`, `mean_price`, `median_price`, `max_price`, `std_price`,
`n_days`, `online_share`, `weekend_share`, `n_txn_365d`, `mean_price_365d`, `n_txn_90d`,
`recency`, `tenure`, `txn_per_active_day`, `repeat_item_ratio`, `spend_per_txn`,
`recent_activity_ratio`, `price_trend`, `log_total_spend`.

**2) Thuộc tính khách hàng:** `active_flag`, `Active`, `club_code`, `news_code`.

**3) Tỷ trọng mua theo category articles ({n_share} chiều):** tỷ lệ lượt mua trong từng nhóm của
`index_group_name`, `index_name`, `garment_group_name`, `section_name`, `product_type_name`
(category hiếm gộp vào `OTHER`).

**4) Target encoding theo fold (9 chiều)** — tuổi trung bình (làm trơn Bayes) của người mua,
tính **chỉ trên tập train của fold**:

| Feature | Ý nghĩa | Alpha |
|---|---|---|
| `te_article` | tuổi TB người mua từng `article_id` (105.542 mức) | 10 |
| `te_department_no` | tuổi TB theo department (627 mức) | 20 |
| `te_product_type_no` | tuổi TB theo product type (647 mức) | 20 |
| `te_index_name` | tuổi TB theo index (11 mức) | 20 |
| `te_section_name` | tuổi TB theo section (594 mức) | 20 |
| `te_product_type_name` | tuổi TB theo tên product type (645 mức) | 20 |
| `art_enc_std` / `art_enc_max` / `art_enc_min` | độ phân tán "độ tuổi" các món đã mua | — |

`postal_code` **đã bị loại**: kiểm định out-of-fold cho thấy nó không có tín hiệu
(MAE 12,05 so với 12,07 của median toàn cục — xem mục 5), chỉ làm nhiễu mô hình.

{eda_summary()}## 4. Kết quả cross-validation ({C.N_SPLITS}-fold, {n_lab:,} khách có nhãn)

| Fold | MAE | RMSE | R² | Đúng nhóm tuổi | Sai ≤ 5 tuổi | Sai ≤ 10 tuổi | Trees |
|---|---|---|---|---|---|---|---|
{rows}

**Tổng hợp out-of-fold:**

| Metric | Giá trị |
|---|---|
| MAE | **{fmt(cv['mae'])} tuổi** |
| RMSE | {fmt(cv['rmse'])} |
| R² | {fmt(cv['r2'], 4)} |
| Đúng nhóm tuổi (7 nhóm) | {pct(cv['bucket_acc'])} |
| Sai ≤ 5 tuổi | {pct(cv['within_5y'])} |
| Sai ≤ 10 tuổi | {pct(cv['within_10y'])} |

## 5. So sánh baseline (cùng out-of-fold)

| Phương pháp | MAE | RMSE | R² | Đúng nhóm tuổi |
|---|---|---|---|---|
| Median toàn cục ({obs['median']:.0f} tuổi) | {fmt(base['global_median']['mae'])} | {fmt(base['global_median']['rmse'])} | {fmt(base['global_median']['r2'], 4)} | {pct(base['global_median']['bucket_acc'])} |
| Postal-code median (OOF) | {fmt(base['postal_median_oof']['mae'])} | {fmt(base['postal_median_oof']['rmse'])} | {fmt(base['postal_median_oof']['r2'], 4)} | {pct(base['postal_median_oof']['bucket_acc'])} |
| **LightGBM (OOF)** | **{fmt(cv['mae'])}** | **{fmt(cv['rmse'])}** | **{fmt(cv['r2'], 4)}** | **{pct(cv['bucket_acc'])}** |

→ Giảm **{(1 - cv['mae'] / base['global_median']['mae']) * 100:.1f}%** MAE so với điền median toàn cục,
giảm **{(1 - cv['mae'] / base['postal_median_oof']['mae']) * 100:.1f}%** so với postal-code median.

## 6. Sai số theo nhóm tuổi (out-of-fold)

| Nhóm tuổi thật | Số khách | Tỷ lệ | MAE | Bias (dự đoán − thật) |
|---|---|---|---|---|
{age_group_table()}

Bias âm ở nhóm cao tuổi và dương ở nhóm trẻ cho thấy mô hình có xu hướng **co về phía trung bình**
(đặc trưng của hồi quy có tín hiệu yếu), nên dự đoán ít cực đoan hơn thực tế.

## 7. Top 25 đặc trưng theo gain

| # | Feature | Gain |
|---|---|---|
{imp_rows}

## 8. Phân phối tuổi

| | Mean | Std | Median | Min | Max |
|---|---|---|---|---|---|
| Quan sát được | {fmt(obs['mean'], 2)} | {fmt(obs['std'], 2)} | {fmt(obs['median'], 1)} | {C.AGE_MIN} | {C.AGE_MAX} |
| Điền (imputed) | {fmt(imp['mean'], 2)} | {fmt(imp['std'], 2)} | {fmt(imp['median'], 1)} | {fmt(imp['min'], 1)} | {fmt(imp['max'], 1)} |

## 9. Cấu hình huấn luyện

```python
num_leaves={p['num_leaves']}, learning_rate={p['learning_rate']}, min_data_in_leaf={p['min_data_in_leaf']},
feature_fraction={p['feature_fraction']}, bagging_fraction={p['bagging_fraction']}, lambda_l2={p['lambda_l2']},
objective="{p['objective']}", metric="{p['metric']}"
```

- Mẫu train mỗi fold: {C.TRAIN_SAMPLE:,} khách (lấy ngẫu nhiên từ tập train của fold).
- Early stopping {C.EARLY_STOPPING} vòng, tối đa {C.NUM_BOOST_ROUND:,} vòng.
- Số cây model cuối cùng: {m['final_num_trees']:,} (best_iter trung bình các fold: {best_iter_mean:.0f}).
- Tổng thời gian: {run_min:.1f} phút.

## 10. Cách chạy

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="."

python -m src.eda            # EDA -> reports/eda_report.md + reports/eda/*.png
python -m src.eda_report
python -m src.prepare_data   # CSV gốc -> parquet gọn
python -m src.features       # feature + ma trận đếm
python -m src.dataset        # X.npy
python -m src.train          # 5-fold CV + model cuối
python -m src.impute         # customers_age_imputed.csv
python -m src.report         # báo cáo này
```

## 11. Lưu ý & hạn chế

- `age` là thông tin **tự nguyện khai báo**, nên nhóm thiếu có thể khác nhóm có nhãn
  (thiên lệch chọn mẫu). EDA cho thấy nhóm thiếu **ít hoạt động hơn** (median 4 so với 9 giao dịch),
  nên MAE kỳ vọng trên nhóm thiếu là **~9,7 tuổi** (xem `reports/eda_report.md` mục 5),
  không phải {fmt(cv['mae'])} như trên nhóm có nhãn.
- Độ chính xác bị chặn trên bởi dữ liệu: hành vi mua chỉ giải thích được ~{cv['r2'] * 100:.0f}% phương sai tuổi
  (R² = {fmt(cv['r2'], 3)}). Các nhóm tuổi gần nhau (ví dụ 25–35) gần như không phân biệt được.
- `age_filled` giữ nguyên giá trị gốc ở các dòng đã có `age`; cột `age_source` cho biết nguồn
  (`observed` / `imputed`). Dự đoán được clip về [{C.AGE_MIN}, {C.AGE_MAX}].
- Không dùng `postal_code` cho target encoding (đã kiểm định là nhiễu).
"""
    C.REPORT_MD.write_text(report, encoding="utf-8")
    print(f"wrote {C.REPORT_MD}")


if __name__ == "__main__":
    main()
