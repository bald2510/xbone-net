# XBone-Net

XBone-Net kết hợp ảnh X-quang và văn bản để phân loại tổn thương xương, phát hiện mẫu ngoài phân phối (OOD) và trực quan hóa Integrated Gradients (IG). Dự án phục vụ nghiên cứu.

- **BTXRD:** phân loại 10 lớp; văn bản đầu vào được sinh từ metadata của bộ dữ liệu.
- **CTCH:** phân loại 22 lớp; văn bản đầu vào là bệnh sử lâm sàng. Demo sử dụng mô hình CTCH, seed 42.

## 1. Bắt đầu từ đâu?

| Bạn muốn làm gì? | Đọc theo thứ tự |
| --- | --- |
| Chạy thử giao diện với mô hình có sẵn | [Cài đặt](HuongDanCaiDat.txt) → [Sử dụng demo](HuongDanSuDungDemo.txt) |
| Huấn luyện và tái lập thí nghiệm | [Cài đặt](HuongDanCaiDat.txt) → [Sử dụng và tái lập](HuongDanSuDung.txt) |

Mọi lệnh đều chạy tại thư mục gốc `xbone-net`, nơi chứa `train.py` và `requirements.txt`. Chạy từng lệnh, kiểm tra kết quả rồi mới sang bước tiếp theo.

## 2. Chạy demo bằng mô hình có sẵn

**Bước 1 — Cài môi trường.** Làm theo [hướng dẫn cài đặt](HuongDanCaiDat.txt), gồm lựa chọn CPU/GPU và kiểm tra thư viện.

**Bước 2 — Chuẩn bị gói model.** Kiểm tra bản mã nguồn được bàn giao có đủ các tệp sau. Nếu thiếu, nhận nguyên gói từ người bàn giao dự án; cài thư viện không tạo ra các tệp này.

```text
demo/model/
├── best_phase2.pth
├── metrics.json
├── manifest.json
└── features/
    ├── ctch_train.npz
    └── ctch_val.npz
```

**Bước 3 — Kiểm tra và khởi động.**

```bash
conda activate Thesis
python tools/verify_demo_artifacts.py
```

Khi thấy `Demo artifacts: OK`, chạy:

```bash
python -m streamlit run demo/streamlit_app.py
```

**Bước 4 — Thử một mẫu.** Mở địa chỉ in trên terminal, thường là `http://localhost:8501`. Chọn ảnh X-quang, nhập bệnh sử tương ứng rồi nhấn **Phân tích**. Xem [hướng dẫn demo](HuongDanSuDungDemo.txt) để đọc kết quả ID/OOD, confidence và IG.

Gói model và pretrained BiomedCLIP tải ở lần đầu cho phép chạy suy luận. Ảnh CTCH tại `data/CTCH/images/` chỉ cần thêm nếu muốn hiển thị ảnh tham chiếu tương tự. Demo không thay thế chẩn đoán y khoa.

## 3. Tái lập thí nghiệm

**Bước 1 — Nhận dữ liệu đã chuẩn bị.** Cần ảnh, văn bản, nhãn và tệp chia tập của đúng phiên bản thí nghiệm. Dữ liệu bệnh nhân không được phân phối cùng mã nguồn công khai. Xem cấu trúc và điều kiện bàn giao trong [hướng dẫn tái lập](HuongDanSuDung.txt).

**Bước 2 — Huấn luyện một seed trên CTCH.**

```bash
python train.py +experiment=ctch/proposed/ours_xbone_net seed=42
```

Kết quả cần có: `checkpoints/ctch/proposed/ours_xbone_net/seed_42/best_phase2.pth`.

**Bước 3 — Đánh giá checkpoint vừa huấn luyện.**

```bash
python evaluate.py +experiment=ctch/proposed/ours_xbone_net seed=42 --bootstrap --n-bootstrap 10000 --save-embeddings
```

Kết quả nằm trong `results/ctch/proposed/ours_xbone_net/seed_42/`, gồm `metrics.json`. Để chạy BTXRD, thay `ctch/proposed/ours_xbone_net` bằng `btxrd/proposed/ours_xbone_net` trong cả hai lệnh.

**Bước 4 — Mở rộng sau khi một seed chạy thành công.** [Hướng dẫn tái lập](HuongDanSuDung.txt) trình bày lần lượt cách chạy ba seed, tổng hợp kết quả, chạy baseline/ablation, đánh giá OOD/IG, đo thời gian và đóng gói model mới.

Giữ nguyên phiên bản mã, dữ liệu chia tập và cấu hình khi đối chiếu kết quả. Chạy lại cùng tên thí nghiệm và seed có thể ghi đè đầu ra; hãy lưu bản cũ trước. Sai khác phần cứng và thư viện có thể làm kết quả số thay đổi.

## 4. Mô hình hoạt động thế nào?

1. Ảnh đi qua phép resize/crop/normalize gốc của BiomedCLIP (`direct_resize`).
2. Phase 1 căn chỉnh biểu diễn ảnh–văn bản bằng LoRA và Semantic Matching Loss.
3. Phase 2 học phân loại bằng cross-attention hai chiều và linear head.
4. Detector OOD kết hợp khoảng cách kNN của ảnh và văn bản. Nó dùng đặc trưng CTCH-train và hiệu chỉnh ngưỡng trên CTCH-validation.
5. IG biểu diễn độ nhạy của dự đoán theo ảnh và token văn bản; đây không phải bản phân vùng tổn thương.

Luồng chính dùng bệnh sử CTCH hoặc văn bản BTXRD sinh từ metadata. Không đưa kết luận chẩn đoán từ báo cáo X-quang vào đầu vào của luồng này.

## 5. Các thư mục cần biết

| Đường dẫn | Nội dung |
| --- | --- |
| `configs/experiment/` | Cấu hình mô hình đề xuất, baseline và ablation |
| `configs/dataset/` | Đường dẫn dữ liệu và thứ tự lớp |
| `data/` | Dữ liệu cục bộ và công cụ kiểm tra dữ liệu |
| `src/` | Mã mô hình, bộ đọc dữ liệu và hàm dùng chung |
| `checkpoints/` | Trọng số sau huấn luyện |
| `results/` | Chỉ số đánh giá, đặc trưng và kết quả phân tích |
| `outputs/` | Log/cấu hình do Hydra sinh ra |
| `demo/` | Giao diện Streamlit và gói model |
| `tools/`, `benchmark/` | Điều phối thí nghiệm và phân tích kết quả |
| `docs/` | Báo cáo, bài báo và tài liệu trình bày |
