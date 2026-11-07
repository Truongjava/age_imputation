# Notebooks — Age Imputation (H&M)

Bộ notebook kể lại toàn bộ bài toán theo thứ tự, **đã chạy sẵn** để xem kết quả/biểu đồ ngay
trên GitHub (không cần tải dữ liệu 3,7 GB để đọc).

| # | Notebook | Nội dung | Thời gian chạy |
|---|---|---|---|
| 0 | [`00_EDA.ipynb`](00_EDA.ipynb) | EDA: missingness, phân phối tuổi, tương quan, tín hiệu article vs postal, ảnh hưởng MNAR | ~40 s |
| 1 | [`01_prepare_data.ipynb`](01_prepare_data.ipynb) | Chuẩn hoá CSV → Parquet gọn (31,8M giao dịch: 3,5 GB → 194 MB) | ~30 s |
| 2 | [`02_features.ipynb`](02_features.ipynb) | Feature engineering: RFM, tỷ trọng category, ma trận khách × sản phẩm | ~3 phút |
| 3 | [`03_target_encoding.ipynb`](03_target_encoding.ipynb) | Target encoding nhiều mức + **kiểm chứng chống rò rỉ nhãn** | ~25 s |
| 4 | [`04_training.ipynb`](04_training.ipynb) | 5-fold CV với LightGBM, baseline, sai số theo nhóm tuổi, feature importance | ~20 s (dùng lại kết quả) / ~27 phút (train lại) |
| 5 | [`05_imputation_report.ipynb`](05_imputation_report.ipynb) | Điền tuổi, kiểm tra file output, sinh báo cáo Markdown | ~10 s |

## Cách chạy

```powershell
python -m pip install -r requirements.txt
```

Sau đó mở notebook theo thứ tự 00 → 05. Mỗi notebook tự thêm thư mục gốc repo vào `sys.path`,
nên mở từ thư mục `notebooks/` hay từ gốc repo đều được.

## Dữ liệu

Cần tải từ Kaggle và đặt **ở thư mục gốc repo**:

```
customers.csv
articles.csv
transactions_train.csv
```

## Ghi chú

- Notebook 4 mặc định `TRAIN_FROM_SCRATCH = False` → đọc lại `outputs/oof_pred.npy` và
  `reports/metrics.json` để hiển thị kết quả tức thì. Đổi thành `True` nếu muốn train lại đầy đủ
  5 fold (cần ~4 GB RAM, ~27 phút trên CPU 6 lõi).
- Các bước 01–02 ghi artifact vào `artifacts/` (đã ignore trong `.gitignore`) để notebook 03–05
  chạy nhanh.
- Báo cáo dạng văn bản: [`reports/eda_report.md`](../outputs/eda_report.md) và
  [`reports/age_imputation_report.md`](../outputs/age_imputation_report.md).
