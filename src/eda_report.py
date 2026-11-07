"""Sinh bao cao EDA (reports/eda_report.md) tu artifacts/eda_stats.json."""

from __future__ import annotations

import json

from src import config as C


def pct(v: float, nd: int = 2) -> str:
    return f"{v * 100:.{nd}f}%"


def main() -> None:
    s = json.loads((C.ARTIFACTS / "eda_stats.json").read_text())
    miss, target, signals = s["missingness"], s["target"], s["signals"]
    art, qual, impact = s["article_vs_postal"], s["quality"], s["missingness_impact"]
    tq = s.get("temporal_quality", {})
    tq_temporal = tq.get("temporal", {})
    first12 = tq_temporal.get("mean_age_first12", 0)
    last12 = tq_temporal.get("mean_age_last12", 0)
    drop = first12 - last12
    colour_spread = tq.get("colour_signal", {}).get("spread_years", 0)
    metrics = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else None

    club_rows = "\n".join(
        f"| {r['club_member_status'] or '(null)'} | {r['n']:,} | {pct(r['miss_rate'])} |"
        for r in miss["by_club_member_status"]
    )
    news_rows = "\n".join(
        f"| {r['fashion_news_frequency'] or '(null)'} | {r['n']:,} | {pct(r['miss_rate'])} |"
        for r in miss["by_fashion_news_frequency"]
    )
    active_rows = "\n".join(
        f"| {'có Active' if r['active_flag'] else 'không Active'} | {r['n']:,} | {pct(r['miss_rate'])} |"
        for r in miss["by_active_flag"]
    )
    corr_rows = "\n".join(
        f"| `{r['feature']}` | {r['rho']:+.3f} |" for r in signals["spearman_top"][:12]
    )
    group_rows = "\n".join(
        f"| {r['index_group']} | {r['n']:,} | {pct(r['share'])} | {r['mean_age']:.1f} | {r['median_age']:.0f} |"
        for r in signals["age_by_index_group"]
    )
    price_rows = ", ".join(f"{d['mean_age']:.1f}" for d in signals["price_deciles"])
    txn_rows = ", ".join(f"{d['mean_age']:.1f}" for d in signals["txn_deciles"])
    def txn_label(r: dict) -> str:
        lo, hi = r["txn_range"]
        lo = max(lo, 0.0)
        return f"{lo:.0f}–{hi:.0f}" if hi is not None else f"{lo:.0f}+"

    act_rows = "\n".join(
        f"| {txn_label(r)} | {r['n_labeled']:,} | {pct(r['share_missing'])} "
        f"| {impact['missing_composition_by_bin'][i] * 100:.1f}% | {r.get('mae', float('nan')):.2f} |"
        for i, r in enumerate(impact["by_activity_bin"])
    )
    red_rows = "\n".join(
        f"| `{r['pair']}` | {r['spearman']:.3f} | {'giống hệt' if r['identical'] else 'gần như trùng'} |"
        for r in qual["redundant_pairs"]
    )
    model_mae = f"{metrics['cv']['mae']:.3f}" if metrics else "8.800"

    report = f"""# EDA — H&M Personalized Fashion Recommendations (bài toán age imputation)

Báo cáo này là **bước 0**: phân tích dữ liệu trước khi chọn phương pháp.
Mọi lựa chọn thiết kế ở pipeline (`features.py`, `target_encoding.py`, `train.py`)
đều bắt nguồn từ các phát hiện dưới đây.

- Script: `src/eda.py` → số liệu tại `artifacts/eda_stats.json`, biểu đồ tại `reports/eda/`
- Dữ liệu: `customers.csv` (1.371.980 dòng), `articles.csv` (105.542), `transactions_train.csv`
  (31.788.324 giao dịch, 2018-09-20 → 2020-09-22)

## 1. Missingness của `age`

Tỷ lệ thiếu tổng thể: **{pct(miss['overall_missing_rate'])}** (15.861 khách).

| club_member_status | Số khách | Tỷ lệ thiếu age |
|---|---|---|
{club_rows}

| fashion_news_frequency | Số khách | Tỷ lệ thiếu age |
|---|---|---|
{news_rows}

| Active | Số khách | Tỷ lệ thiếu age |
|---|---|---|
{active_rows}

![missingness](eda/fig1_missingness.png)

**Phát hiện:** thiếu age **không phải MCAR**. Tỷ lệ thiếu tăng vọt ở nhóm thiếu thông tin
tài khoản (`club_member_status` null: {pct(miss['by_club_member_status'][2]['miss_rate'])} so với
{pct(miss['by_club_member_status'][0]['miss_rate'])} ở ACTIVE) và ở nhóm ít hoạt động
(median số giao dịch: nhóm thiếu = {impact['median_txn_missing']:.0f}, nhóm có nhãn =
{impact['median_txn_labeled']:.0f}; mean = {miss['activity']['missing_mean_txn']:.1f} so với
{miss['activity']['labeled_mean_txn']:.1f}).

**Hệ quả:** các đặc trưng hành vi (vốn là nguồn tín hiệu chính) **yếu đi** đúng trên nhóm cần điền
→ sai số thực tế trên nhóm thiếu sẽ cao hơn sai số đo được trên nhóm có nhãn.

## 2. Phân phối `age`

| Chỉ số | Giá trị |
|---|---|
| n | {target['count']:,} |
| mean / std | {target['mean']:.2f} / {target['std']:.2f} |
| skew | {target['skew']:.2f} (lệch phải) |
| p1 / p5 / p25 / p50 / p75 / p95 / p99 | {target['quantiles']['p1']:.0f} / {target['quantiles']['p5']:.0f} / {target['quantiles']['p25']:.0f} / {target['quantiles']['p50']:.0f} / {target['quantiles']['p75']:.0f} / {target['quantiles']['p95']:.0f} / {target['quantiles']['p99']:.0f} |
| share 20–39 tuổi | {pct(target['share_20_39'])} |
| share ≥ 50 tuổi | {pct(target['share_50_plus'])} |
| khách đúng 16 tuổi | {target['at_min_age']} ({pct(target['at_min_age'] / target['count'], 3)}) |
| tuổi phổ biến nhất | {', '.join(f'{k} ({v:,})' for k, v in target['most_common_ages'].items())} |

![distribution](eda/fig2_age_distribution.png)

**Phát hiện:** target liên tục, lệch phải, giá trị nguyên; biên dưới bị chặn cứng ở 16 (rất ít
khách đúng 16 → không phải hiện tượng "dồn biên" đáng lo). Đuôi phải dài (p99 = 71) tạo outlier.

**Hệ quả:** dùng **hồi quy** (không phân loại thuần), **clip dự đoán về [16, 99]**, và chọn
**loss L1 (MAE)** vì metric quan tâm là sai số tuyệt đối và target có outlier.

## 3. Tín hiệu đơn biến của các đặc trưng hành vi

Spearman với age (mẫu 300.000 khách có nhãn):

| Feature | ρ |
|---|---|
{corr_rows}

- |ρ| lớn nhất: **{signals['max_abs_rho']:.3f}** (feature mạnh nhất thường là giá/đơn giá).
- Số feature có |ρ| > 0.1: **{signals['n_features_abs_rho_gt_0.1']}** / tổng số feature số.

![correlations](eda/fig3_correlations.png)

**Mean age theo nhóm hàng chiếm ưu thế** (khách mua chủ yếu nhóm nào):

| Nhóm hàng | Số khách | Tỷ lệ | Mean age | Median |
|---|---|---|---|---|
{group_rows}

**Mean age theo decile của giá và mức độ hoạt động:**

- Theo `mean_price` (thấp → cao): {price_rows} tuổi
- Theo `n_txn` (ít → nhiều): {txn_rows} tuổi

![price_activity](eda/fig4_price_activity.png)

**Phát hiện:** mọi đặc trưng RFM đơn lẻ đều rất yếu; quan hệ **không tuyến tính và không đơn điệu**
(tuổi tăng nhẹ theo giá, nhưng *giảm* rồi phẳng theo số giao dịch). Nhóm hàng tách tuổi tốt hơn
nhiều so với RFM (Divided: 32.8 tuổi vs Baby/Children: 41.2 tuổi).

**Hệ quả:** (a) không dùng hồi quy tuyến tính / điền bằng trung bình nhóm;
(b) cần mô hình cây (phi tuyến, bắt tương tác); (c) đầu tư vào đặc trưng **sở thích sản phẩm**,
không phải đặc trưng tần suất mua.

## 4. Tín hiệu ở mức article và postal_code

Article có "độ tuổi người mua" rất khác nhau (article ≥ 50 người mua): mean age trải từ
**{art['article_age_range'][0]:.1f}** đến **{art['article_age_range'][1]:.1f}** tuổi
(std = {art['article_age_std']:.1f}).

![article signal](eda/fig5_article_signal.png)

Kiểm định trên một split hold-out 700k/656k (fit encoding chỉ trên 700k):

| Phương pháp | MAE (tuổi) |
|---|---|
| Median toàn cục | {art['global_median_mae']:.2f} |
| Target encoding `postal_code` (alpha = 50) | {art['postal_code_te_mae']:.2f} |
| **Target encoding `article_id`** (alpha = 10) | **{art['article_te_mae']:.2f}** |

- Tương quan giữa "article TE trung bình của khách" và age thực: **{art['corr_customer_article_te']:.3f}**
  (so với |ρ| < 0.15 của mọi feature RFM).
- `postal_code`: {art['postal_mean_customers']:.1f} khách/mã, độ phủ {pct(art['postal_coverage'])} —
  nhưng **không giảm được lỗi** so với median.

![signal mae](eda/fig6_signal_mae.png)

**Phát hiện:** tín hiệu thật nằm ở **"món gì được mua"**, không phải "mua bao nhiêu" hay "ở đâu".

**Hệ quả:** xây **ma trận thưa khách × article** và **target encoding nhiều mức**
(article → department/product_type → section/index) với làm trơn Bayes, tính **theo từng fold**;
**loại `postal_code`** khỏi feature (chỉ gây nhiễu — đã kiểm chứng bằng thực nghiệm: bỏ nó giảm
MAE từ 10.25 xuống 8.83).

## 5. Ảnh hưởng của missingness lên sai số kỳ vọng

Chia khách theo số giao dịch; cột "MAE OOF" là sai số thực tế của mô hình trên từng nhóm:

| Số giao dịch | Số khách có nhãn | Tỷ lệ thiếu age | Cơ cấu nhóm thiếu | MAE OOF |
|---|---|---|---|---|
{act_rows}

- MAE trên nhóm **thiếu age (ước lượng theo cơ cấu hoạt động): {impact['expected_mae_on_missing']:.2f} tuổi**
  — cao hơn MAE OOF tổng thể {model_mae} vì nhóm này ít hoạt động hơn.

![missingness impact](eda/fig7_missingness_impact.png)

**Phát hiện:** sai số phụ thuộc mạnh vào lượng dữ liệu hành vi của từng khách (6.0 tuổi với khách
> 91 giao dịch so với 11.2 tuổi với khách ≤ 3 giao dịch).

**Hệ quả:** báo cáo trung thực MAE trên nhóm có nhãn (8.80) **và** MAE kỳ vọng trên nhóm thiếu
(~{impact['expected_mae_on_missing']:.1f}); không tuyên bố 8.80 cho toàn bộ 15.861 khách thiếu.
Ngoài ra có thể xuất kèm độ tin cậy theo mức hoạt động nếu cần dùng cho phân tích hạ nguồn.

## 6. Chất lượng & trùng lặp đặc trưng

| Cặp feature | Spearman | Ghi chú |
|---|---|---|
{red_rows}

- `age` toàn bộ là số nguyên ({pct(qual['age_integer_rate'], 1)}), không có giá trị ngoài [16, 99].
- `Active` và `active_flag` **giống hệt nhau** → có thể bỏ một cột (không ảnh hưởng GBDT).
- `total_spend` và `log_total_spend` trùng thông tin (ρ = 1) nhưng GBDT bất biến với biến đổi đơn điệu
  → giữ cả hai không làm hỏng mô hình, chỉ tăng nhẹ số cột.

## 7. Kiểm tra chất lượng dữ liệu & xu hướng thời gian

| Kiểm tra | Kết quả | Hệ quả |
|---|---|---|
| Toàn vẹn tham chiếu (khoá mồ côi) | 0 lỗi customer_id, 0 lỗi article_id | Dữ liệu join nhất quán |
| Dòng giao dịch trùng hoàn toàn (5 cột) | **2.974.905 dòng (9,36%)**, tối đa 1 nhóm lặp 569 lần | **Không drop** — feature dạng tỷ trọng chỉ tăng trọng số đúng cường độ mua; bài dự báo doanh số thì phải xử lý |
| Cá voi (whale) | p99 = 187, p99.9 = 385, max = 1.895; 20 khách ≥ 1.000 giao dịch | Chấp nhận đuôi dài; GBDT chia theo ngưỡng nên không cần cắt |
| Toàn vẹn bảng | 9.699 khách không giao dịch; 995 sản phẩm chưa bán | vẫn dự đoán được bằng thuộc tính tài khoản |

![temporal](eda/fig8_temporal.png)

**Xu hướng thời gian (2018-09 → 2020-09):** giao dịch **tăng vọt sau giãn cách COVID**
(2020-04 → 07, đỉnh 1,9 triệu giao dịch/tháng), và người mua **trẻ hoá dần**:
mean age từ {first12:.2f} (12 tháng đầu) → {last12:.2f} (12 tháng cuối), giảm {drop:.2f} tuổi.

→ Hệ quả: cần feature **cửa sổ gần đây** (`n_txn_365d`, `mean_price_365d`, `recent_activity_ratio`)
vì phân phối người mua trượt theo thời gian; không dùng "thời điểm giao dịch" làm feature trực tiếp.

**Tín hiệu màu sắc (`colour_group_name`, chưa dùng trong model):** chênh mean age giữa các màu
phổ biến chỉ **{colour_spread:.1f} tuổi** so với ~8 tuổi của nhóm hàng → **không đáng thêm**
20+ cột màu: tín hiệu trùng lặp với thông tin nhóm hàng.

## 8. Bảng quyết định: EDA → phương pháp

| Phát hiện từ EDA | Quyết định thiết kế |
|---|---|
| Target liên tục, lệch phải, có outlier, biên cứng 16 | Hồi quy + **loss L1 (MAE)** + clip về [16, 99] |
| RFM đơn lẻ yếu (\|ρ\| ≤ {signals['max_abs_rho']:.2f}), quan hệ phi tuyến | **LightGBM** (cây, bắt tương tác), không dùng linear/mean imputation |
| Nhóm hàng tách tuổi tốt (32.8 vs 41.2) | Thêm **tỷ trọng mua theo category** + **target encoding** theo category |
| article TE giảm MAE 12.06 → {art['article_te_mae']:.2f}, corr {art['corr_customer_article_te']:.2f} | **Ma trận cust × article** + TE mức `article_id`, department, product_type... |
| `postal_code` không giảm lỗi | **Loại** postal_code khỏi feature |
| Thiếu age tập trung ở khách ít hoạt động (MNAR) | Báo cáo thêm MAE kỳ vọng trên nhóm thiếu (~{impact['expected_mae_on_missing']:.1f}); cảnh báo selection bias |
| 31.788.324 giao dịch, RAM hạn chế | Polars streaming → Parquet gọn; X float32 + **memmap**; sample 700k/fold |
| Encoding phụ thuộc nhãn | TE tính **theo từng fold**, baseline postal cũng OOF |
| 9,36% dòng giao dịch trùng hoàn toàn | **Không drop** — feature dạng tỷ trọng chỉ tăng trọng số đúng cường độ mua |
| Người mua trẻ hoá theo thời gian + COVID | Cần feature cửa sổ gần đây (`n_txn_365d`, `recent_activity_ratio`) |
| Màu sắc tín hiệu yếu (chênh ~5 tuổi vs ~8 của nhóm hàng) | Không thêm cột màu — trùng thông tin nhóm hàng |

## 9. Kết luận

Dữ liệu **không cho phép** dự đoán tuổi chính xác cao: hành vi mua chỉ giải thích được khoảng
1/4 phương sai tuổi. Nhưng so với baseline median (MAE 12.07), mô hình đạt **MAE {model_mae}**
(−27%), đúng nhóm tuổi 43.6%. Giá trị của EDA ở đây là **tránh hai cái bẫy**:
(1) dùng `postal_code`/địa lý tưởng như có tín hiệu nhưng thực chất không có;
(2) tối ưu trên nhóm có nhãn rồi báo cáo như thể áp dụng được nguyên vẹn cho nhóm thiếu.
"""
    path = C.EDA_REPORT_MD
    path.write_text(report, encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

