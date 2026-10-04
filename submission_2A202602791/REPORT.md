# Báo cáo Lab Day 1 — Tống Trần Tiến Dũng — 2A202602791

## 1. Thiết lập

- Môi trường: Kaggle, PyTorch 2.11.0+cu128, GPU Tesla T4.
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val.
- Model: `M-base` (54→256→128→7, 47 879 tham số). Baseline: cross-entropy, SGD + momentum 0,9, lr = 0,1, batch = 512, 20 epoch, He initialization.
- Mốc tham chiếu: accuracy đoán lớp đa số trên val = 0,4876.
- Các chủ đề đã thử: ☐ loss ☑ hyper-parameter ☑ dropout ☐ clipping ☐ mixed precision ☑ init.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | 2,1552 / 1,9459 |
| Quá khớp 20 mẫu: loss cuối | 1,8824 → 0,000002; accuracy = 100% |
| Mọi tham số có gradient khác 0 | ☑ có |
| Baseline, số seed đã chạy | 3: `base-s1`, `base-s2`, `base-s3` |
| Baseline: val acc (TB ± σ) | 0,9074 ± 0,0005 |
| Baseline: val macro-F1 (TB ± σ) | 0,8524 ± 0,0020 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = 0,0040 (val macro-F1). Đây là ước lượng từ ba seed baseline.

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — CE vs MSE

- Dự đoán: chưa thử MSE.
- Kết quả (`exp_id`; ảnh): chưa có thí nghiệm CE vs MSE.
- Giải thích: baseline dùng cross-entropy với logits thô và nhãn `int64`; chưa đủ bằng chứng để so sánh hai hàm mất mát.

### 3.2 Bộ tối ưu hoá

- Dự đoán: chưa đặt thí nghiệm so sánh các bộ tối ưu.
- Bảng nhỏ: các lần chạy chính đều dùng SGD + momentum (`base-s1`, `base-s2`, `base-s3`, `drop-0.3`, `arch-deep`, `init-xavier`).
- Độ nhạy với lr: chưa có ảnh chồng giữa các optimizer.
- Giải thích: lr = 0,1 được chọn trong bước dò validation của baseline; chưa thể kết luận optimizer nào tốt hơn khi chưa chỉnh lr công bằng cho từng optimizer.

### 3.3 Hyper-parameter

- Yếu tố đã đổi: độ sâu mạng, giữ nguyên loss, optimizer, lr, batch, epoch, seed và validation split.
- `arch-deep` (`54→256→128→64→7`, 55 687 tham số) đạt val macro-F1 = 0,8680 ở epoch 17, cao hơn baseline trung bình 0,8524 khoảng 0,0156 và lớn hơn 2σ = 0,0040.
- Cơ chế phù hợp với dự đoán: thêm một lớp ẩn làm tăng sức biểu diễn, giúp giảm underfitting; thời gian mỗi epoch tăng lên khoảng 1,46 giây.

### 3.4 Dropout

- `drop-0.3` đạt val macro-F1 = 0,7826, thấp hơn `base-s1` = 0,8547 khoảng 0,0721, lớn hơn độ nhiễu seed.
- Train loss cuối = 0,3126 và val loss = 0,3192, khoảng cách 0,0067; baseline `base-s1` có train loss = 0,2136 và val loss = 0,2392, khoảng cách 0,0256.
- Dropout đã thu hẹp khoảng cách train–val nhưng làm cả hai loss tăng, cho thấy baseline không cần regularization mạnh ở cấu hình này.
- Ảnh: `figures/drop-0.3.png`.

### 3.5 Gradient clipping

- Chưa chạy thí nghiệm clipping nên chưa thể kết luận clipping có kích hoạt hay cứu được huấn luyện ở lr cao.

### 3.6 Mixed precision

- Chưa chạy FP16 hoặc BF16; baseline dùng FP32 trên Tesla T4.

### 3.7 Khởi tạo tham số

- `init-xavier` có loss bước 0 = 2,0222, thấp hơn `base-s1` với He = 2,2691.
- Val macro-F1 của `init-xavier` = 0,8539, so với `base-s1` = 0,8547; chênh lệch −0,0008, nhỏ hơn 2σ nên chưa có bằng chứng Xavier cải thiện điểm cuối.
- Cơ chế: Xavier làm thay đổi phương sai logits ban đầu; He phù hợp hơn với các lớp dùng ReLU. Khởi tạo số 0 không được chạy trong nhóm này, nhưng về nguyên tắc sẽ làm các neuron cùng lớp cập nhật đối xứng.
- Ảnh: `figures/init-xavier.png`.

## 4. Đánh giá cuối trên tập eval

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---:|---:|---:|---:|
| Baseline (`base-s1`) | 1 | 0,8547 | 0,8555 | 0,9077 |
| Cấu hình cuối cùng (`arch-deep`) | 1 | 0,8680 | 0,8716 | 0,9167 |

- Cấu hình cuối cùng là `arch-deep`: M-deep, cross-entropy, SGD + momentum 0,9, lr = 0,1, batch = 512, 20 epoch, He initialization, FP32, seed 1. Cấu hình được chọn bằng val macro-F1; eval chỉ dùng sau khi chốt.
- Eval macro-F1 tăng 0,0161 so với baseline, lớn hơn ngưỡng 2σ val = 0,0040. Đây là đối chiếu tham khảo vì độ nhiễu được đo trên val.
- Val và eval gần nhau: baseline lệch khoảng 0,0008; cấu hình cuối lệch khoảng 0,0036. Không có dấu hiệu lệch lớn giữa hai tập.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|---:|---:|---:|---:|
| 0 | 42 368 | 0,9291 | 0,8962 | 0,9124 |
| 1 | 56 661 | 0,9181 | 0,9420 | 0,9299 |
| 2 | 7 151 | 0,8995 | 0,9348 | 0,9168 |
| 3 | 549 | 0,8379 | 0,7723 | 0,8038 |
| 4 | 1 899 | 0,7704 | 0,7936 | 0,7818 |
| 5 | 3 473 | 0,8617 | 0,8071 | 0,8335 |
| 6 | 4 102 | 0,9278 | 0,9183 | 0,9231 |

- Lớp khó nhất là lớp 4 (F1 = 0,7818). Lớp này bị nhầm nhiều nhất thành lớp 1: 331 mẫu.
- Lớp 4 có support thấp hơn các lớp 0–2, nên macro-F1 nhạy với lỗi của lớp này. Một hướng cải thiện là thử loss có trọng số theo lớp hoặc điều chỉnh sampling, sau đó chọn lại chỉ bằng validation.

Ma trận nhầm lẫn của cấu hình cuối (hàng = thật, cột = dự đoán):

```text
[[37969  4086     0     3    40    11   259]
 [ 2555 53372   153     0   378   169    34]
 [    1   156  6685    51    22   236     0]
 [    0     0   105   424     0    20     0]
 [   27   331    20     0  1507    14     0]
 [    8   156   469    28     9  2803     0]
 [  305    30     0     0     0     0  3767]]
```

## 5. Trả lời các câu hỏi dẫn dắt

1. Chưa thử nhiều bộ tối ưu nên chưa thể kết luận optimizer nào thắng khi chỉnh lr công bằng. Kết quả hiện tại chỉ xác nhận SGD + momentum với lr = 0,1 là baseline đã chạy.
2. Dropout 0,3 làm giảm khoảng cách train–val nhưng làm val macro-F1 giảm mạnh. Vì vậy dropout nên dùng khi có bằng chứng quá khớp, không nên mặc định thêm khi mô hình còn underfit.
3. Gradient clipping giới hạn norm gradient trước bước cập nhật để tránh bước nhảy quá lớn. Thí nghiệm này chưa bật clipping nên chưa có quan sát thực nghiệm về ngưỡng cắt.
4. Mixed precision chưa được thử; chưa thể kết luận về tốc độ hoặc bộ nhớ.
5. Khởi tạo toàn số 0 làm các neuron có cập nhật đối xứng và không học được các đặc trưng khác nhau. He dùng phương sai lớn hơn Xavier, phù hợp với mạng ReLU; Xavier giữ phương sai cân bằng hơn giữa vào và ra.
6. Nếu loss không giảm sau 2 000 bước, ba kiểm tra đầu tiên là: kiểm tra shape/dtype và miền nhãn; đo loss bước 0/logits để phát hiện lỗi loss; kiểm tra gradient hữu hạn, khác 0 và learning rate qua đường cong train/val.

## 6. Hạn chế và điều bất ngờ

- Kết quả dropout khác dự đoán: khoảng cách train–val giảm nhưng điểm validation giảm, cho thấy baseline chưa quá khớp.
- Chỉ có ba seed baseline và mỗi chủ đề một cấu hình; 2σ là ước lượng thô. Chưa thử loss MSE, optimizer khác, clipping hoặc mixed precision.
- Cấu hình cuối chỉ được chạy eval với seed 1. Nếu có thêm thời gian, nên lặp lại `arch-deep` với nhiều seed và thử loss có trọng số cho các lớp ít mẫu.

## 7. Phụ lục

- Danh sách file đã nộp: `code/lab.ipynb`, `code/*.py`, `figures/`, `results/`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `REPORT.md`.
- Thời gian chạy: mỗi epoch baseline khoảng 1,35 giây; `arch-deep` khoảng 1,46 giây trên Tesla T4, chưa tính thời gian chuẩn bị dữ liệu và các thí nghiệm khác.
