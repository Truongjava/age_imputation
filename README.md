# Age Imputation — H&M Personalized Fashion Recommendations

Điền `age` còn thiếu trong `customers.csv` bằng LightGBM, dựa trên hành vi mua sắm
(31.788.324 giao dịch) kết hợp thuộc tính tài khoản.

## Kết quả (5-fold out-of-fold, 1.356.119 khách có nhãn)

| Phương pháp | MAE | RMSE | R² | Đúng nhóm tuổi |
|---|---|---|---|---|
| Median toàn cục (32 tuổi) | 12.067 | 14.971 | −0.094 | 17.3% |
| Postal-code median (OOF) | 12.050 | 14.790 | −0.068 | 17.6% |
| **LightGBM (105 features)** | **8.800** | **12.296** | **0.262** | **43.6%** |

- Sai ≤ 5 tuổi: 46.1% · Sai ≤ 10 tuổi: 68.2%
- 15.861 khách được điền tuổi: mean 37.7, std 9.3, khoảng 17.8–68.6.

`postal_code` không mang tín hiệu (MAE 12.05 ≈ 12.07 của median) nên **không** dùng làm feature.

## Notebooks (đã chạy sẵn, xem được ngay trên GitHub)

Thư mục [`notebooks/`](notebooks) kể lại toàn bộ quá trình, có ghi chú giải thích từng bước và
biểu đồ đã nhúng sẵn:

| # | Notebook | Nội dung |
|---|---|---|
| 0 | [`00_EDA.ipynb`](notebooks/00_EDA.ipynb) | EDA: missingness (MNAR), phân phối tuổi, tương quan, tín hiệu article vs postal |
| 1 | [`01_prepare_data.ipynb`](notebooks/01_prepare_data.ipynb) | Chuẩn hoá CSV → Parquet (3,5 GB → 194 MB) |
| 2 | [`02_features.ipynb`](notebooks/02_features.ipynb) | RFM + tỷ trọng category + ma trận khách × sản phẩm |
| 3 | [`03_target_encoding.ipynb`](notebooks/03_target_encoding.ipynb) | Target encoding nhiều mức + kiểm chứng chống rò rỉ nhãn |
| 4 | [`04_training.ipynb`](notebooks/04_training.ipynb) | 5-fold CV, baseline, sai số theo nhóm tuổi, feature importance |
| 5 | [`05_imputation_report.ipynb`](notebooks/05_imputation_report.ipynb) | Điền tuổi, kiểm tra output, sinh báo cáo |

## Quy trình: EDA trước, rồi mới chọn phương pháp

`src/eda.py` chạy **trước** pipeline và trả lời: thiếu age có ngẫu nhiên không, phân phối target
ra sao, đặc trưng nào thực sự có tín hiệu, feature nào trùng lặp. Kết quả ghi ở
`reports/eda_report.md` + `reports/eda/*.png`. Các kết luận chính:

| Phát hiện | Số liệu | Quyết định |
|---|---|---|
| Thiếu age **không ngẫu nhiên** (MNAR) | 37.7% ở nhóm `club_member_status` null vs 0.49% ACTIVE; median giao dịch 4 vs 9 | Báo cáo thêm MAE kỳ vọng trên nhóm thiếu (~9.7) |
| Target lệch phải, đuôi dài | skew 0.61, p99 = 71, min 16 | Hồi quy + loss L1 + clip [16, 99] |
| RFM đơn biến rất yếu | max \|ρ\| = 0.148 | Không dùng linear; dùng GBDT + đặc trưng sở thích |
| Tín hiệu nằm ở **article** | article TE MAE 11.41 (median 12.06), corr 0.49 | Ma trận cust × article + TE nhiều mức theo fold |
| `postal_code` vô tín hiệu | MAE 12.06 ≈ median 12.06 | Loại khỏi feature |
| Sai số phụ thuộc mức hoạt động | MAE 6.0 (>91 txn) → 11.2 (≤3 txn) | Không quảng bá 8.8 cho toàn bộ nhóm thiếu |

## Cấu trúc

```
data/                 # CSV gốc từ Kaggle (~4 GB, ignored — xem data/README.md)
notebooks/            # 6 notebook kể lại toàn bộ quá trình (đã chạy sẵn)
  00_EDA.ipynb ... 05_imputation_report.ipynb
src/                  # pipeline dạng script (dùng chung logic với notebook)
  config.py           # đường dẫn, tham số, danh sách category
  eda.py              # EDA: missingness, phân phối, tương quan, article vs postal
  eda_report.py       # xuất reports/eda_report.md
  prepare_data.py     # CSV gốc -> parquet gọn (giao dịch dạng số nguyên)
  features.py         # RFM + tỷ trọng category + ma trận cust x article
  dataset.py          # ma trận X float32 ghi ra .npy (memmap)
  target_encoding.py  # target encoding theo fold (article/department/...)
  train.py            # 5-fold CV, LightGBM L1, dự đoán, baseline OOF
  impute.py           # xuất outputs/customers_age_imputed.csv
  report.py           # xuất reports/age_imputation_report.md
reports/              # báo cáo + biểu đồ + metrics (commit lên GitHub)
  eda/                # 7 biểu đồ EDA
artifacts/            # trung gian (ignored)
outputs/              # file kết quả lớn, sinh lại được (ignored)
```

## Chạy

### Cách 1 — Notebook (khuyến nghị để đọc/hiểu)

Xem [`notebooks/README.md`](notebooks/README.md): mở lần lượt `00_EDA` → `05_imputation_report`.

### Cách 2 — Script

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH="."

python -m src.eda            # EDA (~2 phút) -> reports/eda_report.md, reports/eda/*.png
python -m src.eda_report
python -m src.prepare_data   # ~1 phút  -> transactions_compact.parquet
python -m src.features       # ~2 phút  -> base features + ma trận đếm
python -m src.dataset        # ~1 phút  -> X.npy (1.37M x 96)
python -m src.train          # ~27 phút -> CV + model cuối + dự đoán
python -m src.impute         # ghi customers_age_imputed.csv
python -m src.report         # ghi báo cáo Markdown
```

## Kết quả đầu ra

`outputs/customers_age_imputed.csv`:

| Cột | Ý nghĩa |
|---|---|
| `customer_id` | khoá khách hàng (giữ nguyên thứ tự file gốc) |
| `age_original` | tuổi gốc, null nếu vốn thiếu |
| `age_filled` | tuổi sau khi điền (giá trị gốc nếu đã có) |
| `age_source` | `observed` hoặc `imputed` |

`reports/age_imputation_report.md` — báo cáo đầy đủ (kiến trúc, đặc trưng, CV theo fold,
baseline, sai số theo nhóm tuổi, feature importance).

## Thiết kế chống rò rỉ

- Target encoding cho `article_id`, `department_no`, `product_type_no`, `index_name`,
  `section_name`, `product_type_name` được tính **riêng cho từng fold**, chỉ từ khách hàng
  có nhãn thuộc tập train của fold đó; làm trơn Bayes (alpha 10–20).
- Baseline postal-code median cũng tính out-of-fold để so sánh công bằng.
- Metric báo cáo là out-of-fold trên toàn bộ nhóm có nhãn.

## Hạn chế

- `age` là khai báo tự nguyện → nhóm thiếu khác nhóm có nhãn (MNAR): nhóm thiếu ít hoạt động hơn
  (median 4 vs 9 giao dịch) nên **MAE kỳ vọng trên nhóm thiếu ≈ 9.7 tuổi**, cao hơn 8.80 đo trên
  nhóm có nhãn (xem `reports/eda_report.md` mục 5).
- Trần chính xác do dữ liệu: hành vi mua giải thích ~26% phương sai tuổi; các nhóm tuổi
  liền kề (25–35) gần như không phân biệt được. Dự đoán co về trung bình — đây là hành vi
  tối ưu cho MAE (isotonic calibration làm MAE tăng lên 9.35, nên không dùng).
