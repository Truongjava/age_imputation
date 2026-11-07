# Q&A chuẩn bị bảo vệ — Age Imputation trên bộ dữ liệu H&M

Tài liệu ôn tập để hỏi đáp với người hướng dẫn. Mọi số liệu đều lấy từ thực nghiệm trong repo này
(`reports/metrics.json`, `reports/eda_report.md`, `reports/age_imputation_report.md`).

---

## Phần A — Bản chất dữ liệu

### A1. Bộ dữ liệu này nói về gì? Ai dùng, để làm gì?
Bộ [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
(Kaggle, 2022) chứa 2 năm lịch sử mua sắm của H&M: 1.371.980 khách, 105.542 sản phẩm,
31.788.324 giao dịch (2018-09-20 → 2020-09-22). Mục đích gốc của cuộc thi là **gợi ý sản phẩm**;
tôi mượn nó cho bài **age imputation** vì `customers.age` thiếu 1,16% (15.861 khách).

### A2. Ba bảng liên kết thế nào?
- `customers` 1—N `transactions` qua `customer_id` (chuỗi hash 64 ký tự)
- `articles` 1—N `transactions` qua `article_id` (10 chữ số)

Sản phẩm (`articles`) có hệ thống phân loại **phân tầng**: index_group (5+1 nhóm lớn)
→ index (11) → section (594) → department (627) → garment_group (32) → product_type (647).
Càng xuống sâu càng chi tiết — tôi dùng chính phân tầng này cho target encoding nhiều mức.

### A3. Những "cái bẫy" dữ liệu cần biết?
| Bẫy | Con số | Hệ quả |
|---|---|---|
| `customer_id`, `postal_code` là chuỗi hash | 352.899 mã postal | **không có ý nghĩa địa lý** — chỉ là khoá nhóm |
| `price` scale lạ | median 0,025, max 0,59 | khả năng là **tỷ lệ/đơn vị đã chuẩn hoá**, không phải euro → chỉ dùng tương đối, không diễn giải "5 euro" |
| `sample_submission.csv` 270 MB | | file của cuộc thi gốc, **bỏ** — không liên quan age imputation |
| Mật độ thưa | 27,3M cặp / 143 tỷ ô = 0,019% | ma trận khách×sản phẩm phải dùng **sparse** |
| Outlier hoạt động | 9.699 khách 0 giao dịch; 995 sản phẩm chưa bán | vẫn phải dự đoán cho họ, nhưng tài nguyên: chỉ có thuộc tính tài khoản |
| Kênh bán | 70,4% online, 29,6% offline | chỉ 2 giá trị → feature `online_share` |
| Thời gian | đúng 734 ngày, không có gap | dữ liệu hoàn chỉnh về thời gian |
| Dòng trùng hoàn toàn | **2.974.905 dòng (9,36%)**, max 1 nhóm lặp 569 lần | **không drop** cho age imputation — feature dạng tỷ trọng chỉ tăng trọng số đúng cường độ mua; bài dự báo doanh số thì phải xử lý |
| Khoá mồ côi | 0 lỗi customer_id, 0 lỗi article_id | tham chiếu toàn vẹn, join an toàn |
| Whale | 20 khách ≥ 1.000 giao dịch (max 1.895); p99 = 187 | đuôi dài — chấp nhận, GBDT chia theo ngưỡng |
| Xu hướng thời gian | mean age người mua 36,6 → 35,5 (giảm ~1,1 tuổi); giao dịch tăng vọt 2020-04→07 (COVID) | cần feature cửa sổ gần đây; không dùng thời điểm giao dịch trực tiếp |
| Màu sắc | chênh mean age giữa các màu chỉ ~5,4 tuổi (vs ~8 của nhóm hàng) | không đáng thêm cột màu — trùng thông tin nhóm hàng |

### A4. `age` thiếu như thế nào — có phải ngẫu nhiên không?
**Không — MNAR** (Missing Not At Random), bằng chứng:
- Theo `club_member_status`: null → **37,7%** thiếu age; ACTIVE → **0,49%**
- Theo `fashion_news_frequency`: null → 13,8%; Regularly → 0,25%
- Khách không `Active` (không online 30 ngày) → 1,62% so với có `Active` → 0,25%
- Nhóm thiếu age có **median 4 giao dịch** so với **9** của nhóm có nhãn

→ Người ít tương tác hệ thống thì vừa không khai báo tuổi vừa mua ít ⇒ sai số đo trên nhóm có
nhãn (8,80) là **lạc quan** so với nhóm cần điền. Tôi ước lượng MAE kỳ vọng trên nhóm thiếu là
**~9,69 tuổi** bằng trộn theo cơ cấu hoạt động.


### A5. Dữ liệu có "sạch" không — trùng lặp, khoá sai, giá trị lạ?
- **Toàn vẹn tham chiếu:** mọi `customer_id`/`article_id` trong transactions đều tồn tại ở bảng
  cha (0 khoá mồ côi) → join an toàn.
- **Trùng lặp:** 9,36% dòng trùng cả 5 cột (max một nhóm lặp 569 lần). Quyết định: **giữ nguyên**
  vì feature ở dạng tỷ trọng/trung bình — dòng trùng chỉ tăng trọng số đúng cường độ mua. Nếu
  làm bài dự báo doanh số thì phải đổi cách xử lý.
- **Whale:** 20 khách ≥ 1.000 giao dịch. Feature gốc tuyệt đối (`n_txn`) bị whale kéo, nên tôi
  dùng thêm feature tỷ lệ (`txn_per_active_day`, `repeat_item_ratio`).
- **Xu hướng thời gian:** người mua trẻ hoá (mean age ≈ 37 → ≈ 35) và giao dịch tăng vọt sau
  giãn cách COVID 2020. Vì thế feature "cửa sổ gần đây" (`n_txn_365d`, `recent_activity_ratio`)
  quan trọng; không đưa ngày tháng vào feature trực tiếp.

### A6. COVID có làm lệch kết quả không?
Có ảnh hưởng nhưng kiểm soát được: giãn cách 2020 kéo khách trẻ vào (online tăng). Vì tôi dùng
**toàn bộ lịch sử 2 năm** làm feature (không chỉ 2020), và encoding là *tuổi người mua mỗi món*
(nhóm hàng trẻ/t già tách nhau ổn định qua các năm), xu hướng thời gian không làm sai lệch lớn.
EDA (mục 7 của `eda_report.md`) có biểu đồ minh hoạ.

### A7. Phân phối `age`?
Lệch phải (skew ≈ 0,61): mean 36,39, median 32, p99 = 71. Đỉnh ở 20–26 tuổi (56% khách 20–39).
Bị chặn dưới ở 16 (chỉ 161 khách đúng 16 → không phải dồn cục ở biên), 99 hiếm (13 khách). Giá trị nguyên.

### A8. Tại sao nhóm hàng nói được nhiều về tuổi?
Sản phẩm có "độ tuổi người mua" đặc trưng: khách mua chủ yếu `Baby/Children` → mean age 41,2;
`Divided` (dòng trẻ) → 32,8; `Ladieswear` → 36,7; `Menswear` → 38,6; `Sport` → 37,1.
Ở mức sản phẩm: mean buyer age trải 29,8–45,9 (sản phẩm ≥50 người mua) — khoảng 16 tuổi chênh.

---

## Phần B — Phương pháp

### B1. Vì sao không điền median/mean theo nhóm là đủ?
Baseline median toàn cục: MAE 12,07. Postal-code median (out-of-fold): MAE 12,05.
Cả hai **không giảm được lỗi** — và tôi chứng minh `postal_code` vô tín hiệu (A1: chuỗi hash
không ngữ nghĩa). Mô hình LightGBM đạt 8,80 → giảm 27%.

### B2. Tại sao chọn hồi quy + loss L1, không phải phân loại nhóm tuổi?
- EDA: target liên tục, lệch phải, có outlier (p99=71) → **L1 bền với outlier** và tối ưu đúng
  metric quan tâm (sai số tuyệt đối).
- Phân loại 7 nhóm tuổi vứt bỏ thông tin khoảng cách (sai 29→32 không tệ bằng 29→61);
  cũng khó chọn ngưỡng ranh giới.
- Đã cân nhắc và từ chối: isotonic calibration làm MAE **tăng** 8,80 → 9,35 (dự đoán co về
  trung bình là hành vi tối ưu cho MAE).

### B3. Cấu trúc feature 105 chiều?
1. **Hành vi giao dịch** (24): RFM + xu hướng — `n_txn`, `mean_price`, `recency`, `online_share`,
   `price_trend` (giá 1 năm gần / toàn bộ)...
2. **Thuộc tính tài khoản** (4): `club_code`, `news_code`, `Active`, `active_flag`
3. **Tỷ trọng mua theo category** (72): tỷ lệ lượt mua trong từng `index_name`, `section_name`,
   `product_type_name`... (top-K, category hiếm gộp `OTHER`)
4. **Target encoding theo fold** (9): `te_article` (105.542 mức, α=10), `te_department_no`,
   `te_section_name`, `te_product_type_no`... (α=20), cùng `art_enc_std/max/min` (độ phân tán
   "độ tuổi" các món đã mua)

Importance khớp EDA: 6/10 feature mạnh nhất thuộc nhóm 4.

### B4. Target encoding là gì và nó nguy hiểm chỗ nào?
Dùng **chính nhãn** để tạo feature: `enc(c) = (Σ age của người mua c trong train + α·prior) / (n_c + α)`
(làm trơn Bayes, α kéo giá trị hiếm về prior).

**Nguy hiểm:** tính trên toàn bộ dữ liệu rồi cross-validation → nhãn của khách validation đã
thấm vào feature của chính họ → điểm CV đẹp giả.

**Cách tôi chống lại:** trong từng fold, chỉ dùng nhãn của khách thuộc **tập train** của fold đó
để tính encoding. Chứng minh bằng thực nghiệm (notebook 03): đổi nhãn của 200 khách validation
thành 99 → giá trị encoding của họ **không đổi**.

### B5. Vì sao dùng LightGBM?
- Feature hành vi có quan hệ **phi tuyến + không đơn điệu** với tuổi (tuổi tăng nhẹ theo giá
  decile nhưng **giảm** theo số giao dịch) → mô hình cây tự nhiên.
- 1,37 triệu dòng, 105 chiều, chạy CPU 27 phút — đúng vùng GBDT mạnh.
- LightGBM xử lý tốt feature số thưa/one-hot (chia nhánh theo ngưỡng) nên 72 cột tỷ trọng
  category dùng trực tiếp không cần giảm chiều.

### B6. Cấu hình LightGBM?
`num_leaves=63, lr=0.03, min_data_in_leaf=300, feature_fraction=0.7, bagging=0.8, λ=1.0`,
objective `regression_l1`. Mẫu train mỗi fold 700.000 (từ 1,08M khả dụng), early stopping 150,
max 4.000 vòng (best_iter trung bình ~3998). Đã thử tăng mẫu lên 1,08M → cải thiện < 0,01 MAE,
không đáng chi phí.

### B7. Cross-validation thế nào?
5-fold KFold (shuffle, seed=42) trên 1.356.119 khách có nhãn. Encoding được tính **lại 5 lần**
(mỗi fold một lần). Metric báo cáo là **out-of-fold** — mỗi khách được dự đoán bởi model chưa
từng thấy nhãn của họ.

---

## Phần C — Kết quả & đánh giá

### C1. Kết quả chính?
| Phương pháp | MAE | R² | Đúng nhóm tuổi |
|---|---|---|---|
| Median toàn cục (32 tuổi) | 12.067 | −0.094 | 17.3% |
| Postal-code median (OOF) | 12.050 | −0.068 | 17.6% |
| **LightGBM (105 features)** | **8.800** | **0.262** | **43.6%** |

Sai ≤5 tuổi: 46,1% · sai ≤10 tuổi: 68,2% · 5 fold ổn định (MAE 8.78–8.81).

### C2. MAE 8,8 tuổi — lớn thế thì dùng làm gì?
- Là bài toán **h ceiling thấp**: hành vi mua giải thích ~26% phương sai tuổi. Không thể tách
  25 vs 35 tuổi mua cùng đồ nữ.
- Nhưng so với baseline vẫn giảm 27%, và đúng nhóm tuổi 43,6% (baseline 17,3%) — tăng ~2,5 lần.
- **Đúng mục đích**: điền để *bảo toàn phân phối* cho phân tích hạ nguồn, không để suy luận
  từng cá nhân. Nếu cần cá nhân hoá chính xác thì dữ liệu này không đủ.

### C3. R² = 0,26 có phải mô hình kém không?
Cần tách **mô hình kém** khỏi **dữ liệu ceiling thấp**. Bằng chứng ceiling thấp:
- Mọi feature đơn biến |Spearman| < 0,15
- Feature mạnh nhất (`te_article`, target encoding ở mức chi tiết nhất) cũng chỉ corr 0,49
- Tăng mẫu train, tăng vòng boosting đều không cải thiện đáng kể

Baseline R² âm (−0.09) vì phân phối lệch: median tối ưu cho L1 nhưng sai lệch lớn theo RMSE.
Mô hình đẩy lên 0,26 — khoảng cách so với ceiling là điều quan trọng, không phải số tuyệt đối.

### C4. Sai số trải đều theo nhóm tuổi?
Không — thiên lệch **co về trung bình**:

| Nhóm tuổi thật | Tỷ lệ | MAE | Bias (dự đoán − thật) |
|---|---|---|---|
| 16-19 | 5.3% | 7.79 | **+7.78** |
| 20-29 | 39.0% | 5.77 | +4.92 |
| 30-39 | 17.3% | 5.79 | +0.70 |
| 40-49 | 15.1% | 9.18 | −6.56 |
| 50-59 | 16.7% | 14.22 | −13.57 |
| 60-69 | 5.4% | 18.73 | −18.68 |
| 70+ | 1.4% | 27.31 | −27.31 |

Nhóm 70+ bị dự đoán thấp trung bình 27 tuổi (vì chỉ 1,4% dữ liệu thuộc nhóm này).
Tôi đã cân nhắc: không hiệu chỉnh để kéo về (isotonic tăng MAE tổng), nhưng **công khai bảng
này** để người dùng biết giới hạn theo nhóm.

### C5. File kết quả trông thế nào?
`outputs/customers_age_imputed.csv` (110 MB, không commit): 4 cột — `customer_id`,
`age_original` (null nếu vốn thiếu), `age_filled`, `age_source` (`observed` / `imputed`).
11 nhóm khách quan sát không đổi; 15.861 dòng imputed (mean 37,7, std 9,3, khoảng 17,8–68,6).

### C6. Kiểm tra chất lượng output?
Notebook 05 có 5 `assert`: đủ 1.371.980 dòng; giữ thứ tự `customer_id`; tuổi gốc không đổi;
đúng 15.861 dòng `imputed`; không có null/ngoài khoảng [16,99].

---

## Phần D — Câu hỏi khó người hướng dẫn hay hỏi (kèm trả lời)

**D1. "Sao không dùng deep learning / recommender model?"**
GBDT với tabular feature khai thác được gần hết tín hiệu: tăng vòng boosting/tăng mẫu train
cải thiện < 0,01 MAE. Deep model trên ma trận thưa 0,019% cần đầu tư lớn để có thể chỉ ngang.
LightGBM + target encoding chính là kỹ thuật SOTA cho dạng bài này (tương tự các giải pháp
top Kaggle tabular).

**D2. "Sao chắc postal_code vô tín hiệu, có khi bạn làm sai?"**
Đã kiểm định cẩn thận 2 lần, 2 cách khác nhau: (1) target encoding trên hold-out 700k/656k —
MAE 12,06 so với 12,06 của median; (2) khi đưa vào LightGBM như feature, **bỏ nó ra giảm MAE
từ 10,25 xuống 8,83** (nó gây nhiễu). Ngoài ra `postal_code` là chuỗi hash 1 chiều — cũng không
thể tra bảng mapping sang tên thành phố.

**D3. "Bạn có rò rỉ dữ liệu ở đâu không?"**
Nguy cơ lớn nhất là target encoding — đã chống bằng cách tính lại mỗi fold. Các feature khác
không dùng nhãn. Baseline cũng tính OOF. Kiểm chứng bằng thực nghiệm (B4).

**D4. "8,8 tuổi là số chính xác cho 15.861 khách được điền?"**
**Không** — số đó đo trên nhóm có nhãn. Nhóm thiếu ít hoạt động hơn (MNAR). Tôi ước lượng MAE
kỳ vọng trên nhóm thiếu là **~9,69 tuổi** bằng trộn sai số theo cơ cấu hoạt động của chính nhóm đó.

**D5. "Vì sao không thử XGBoost/CatBoost/Random Forest?"**
Chọn LightGBM vì nhỏ gọn, nhanh trên CPU. Về nguyên lý, XGBoost/CatBoost thường cho kết quả
ngang nhau trên tabular (chênh < 2%). Điểm cải tiến không nằm ở thuật toán mà ở feature engineering
— tôi ưu tiên thời gian vào article-level target encoding, chỗ mang lại giảm 1,4 tuổi so với
cùng pipeline không có nó.

**D6. "Có thể cải thiện gì tiếp?"**
1. Feature mô tả sản phẩm (`detail_desc`) → TF-IDF/embedding, có thể tách nhóm tuổi 20–30 tốt hơn.
2. Thời gian: mùa khách mua (đồ mùa đông vs hè) có thể mang tín hiệu nhẹ.
3. Mô hình survival/mixture để mô hình hoá phân phối thay vì điểm dự đoán — giải quyết việc
   co về trung bình ở nhóm 70+.
4. Gán **khoảng tin cậy** thay vì điểm ước lượng: dự đoán phân phối hậu nghiệm theo mức hoạt động.

**D7. "Bạn đã thử hiệu chỉnh (calibration) dự đoán chưa?"**
Có — đã thử isotonic calibration trên dự đoán OOF: MAE tăng 8,80 → 9,35 vì dự đoán co về trung
bình chính là tối ưu cho L1 loss. Ghi nhận là quyết định có ý thức, không phải bỏ qua.


**D8. "Dữ liệu có 9% dòng trùng — bạn xử lý sao?"**
Đã lượng hoá (2.974.905 dòng trùng cả 5 cột). Quyết định **giữ nguyên** vì lý do bài học:
feature của tôi ở dạng *tỷ trọng mua* và *target encoding có trọng số*, dòng trùng tăng cường độ
mua của món đó — đúng thực tế hành vi, không phải nhiễu. Kiểm tra riêng cho bài dự báo doanh số
thì phải dedup. Cá voi (20 khách > 1.000 giao dịch) cũng không cắt: GBDT chia theo ngưỡng nên
tự chống được đuôi dài.

---

## Phụ lục — nơi tra cứu

| Chủ đề | File |
|---|---|
| EDA chi tiết + 7 biểu đồ | `reports/eda_report.md` + `reports/eda/*.png` |
| Báo cáo mô hình (CV, baseline, importance) | `reports/age_imputation_report.md` |
| Toàn bộ số liệu máy đọc | `reports/metrics.json` |
| Notebook có giải thích từng bước | `notebooks/00_EDA.ipynb` → `05_imputation_report.ipynb` |
