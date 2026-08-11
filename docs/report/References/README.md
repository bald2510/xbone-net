# Phân loại tài liệu tham khảo dạng PDF

Các tệp PDF được phân loại dựa trên cây nội dung đang hoạt động của
`main.tex`. Mỗi bài chỉ được lưu tại một vị trí để tránh trùng lặp.

## `Selected`

Thư mục này chỉ chứa các tài liệu được trích dẫn trực tiếp trong Chương 3 -
Phương pháp đề xuất. Các citation hiện có gồm: `[5]`, `[8]`, `[13]`, `[20]`,
`[21]`, `[28]`, `[32]`, `[38]`, `[39]`, `[40]`, `[43]` và `[47]`.

Các tài liệu này tương ứng với hàm mất mát cân bằng lớp, Vision Transformer,
LoRA, ước lượng hiệp phương sai Ledoit-Wolf, Mahalanobis OOD, Laplacian và độ
sắc nét cục bộ, entropy Shannon, mạng nguyên mẫu, Integrated Gradients,
MedCLIP và BiomedCLIP.

## `ByCitation`

- `Scratch`: mô hình huấn luyện từ đầu, kiến trúc nền, dataset và công cụ chú
  thích không thuộc các nhóm chuyên biệt.
- `Med-VLMs`: mô hình thị giác-ngôn ngữ y khoa và các phương pháp thích nghi
  mô hình nền tảng cho tác vụ y khoa.
- `Med-MLLLM`: mô hình ngôn ngữ lớn đa phương thức y khoa và các kiến trúc nền
  trực tiếp cho nhóm mô hình này.
- `OOD Detection`: phát hiện dữ liệu ngoài phân phối, hiệu chỉnh độ tin cậy và
  hình học biểu diễn phục vụ đánh giá OOD.
- `Explainable AI`: mô hình dùng attention hoặc gióng hàng đa phương thức để
  hỗ trợ giải thích dự đoán.
- `Metrics`: tài liệu định nghĩa hoặc phân tích các độ đo, hiệu chỉnh xác suất,
  benchmark hiệu quả và chỉ số đánh giá hình học biểu diễn.

Mỗi bài chỉ được lưu ở một nhóm. Các chương có citation và lý do phân loại
được ghi trong `classification.csv`.

## Tệp danh mục

- `classification.csv`: nhóm, đường dẫn mới và các chương đang cite từng bài.
- `sources.csv`: URL nguồn và metadata của 46 PDF đã tải.
- `missing.csv`: tài liệu chưa lấy được PDF hợp lệ; hiện là citation `[6]`.
- `selection_summary.json`: số lượng PDF theo từng nhóm.
