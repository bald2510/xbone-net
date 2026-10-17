# Mô tả bộ dữ liệu BTXRD

## 1. Tổng quan

| Đặc điểm | Mô tả |
| --- | --- |
| Nguồn dữ liệu | Bộ dữ liệu u xương công khai |
| Dữ liệu ảnh | Ảnh X-quang xương, định dạng `.jpeg` |
| Số mẫu phân lớp | 3.746 |
| Số lớp đích | 10 |
| Bài toán trong dự án | Phân lớp đa lớp, mỗi ảnh có một nhãn đích |
| Văn bản đi kèm | Văn bản tiếng Anh tổng hợp từ siêu dữ liệu |
| Huấn luyện / xác thực / kiểm thử | 2.621 / 375 / 750 |
| Đơn vị chia tập | Ảnh |
| Tập ngoài phân phối riêng | Không có bảng OOD riêng trong thư mục BTXRD đang kiểm tra |
| Vai trò nghiên cứu | Khảo sát phân loại u xương trong thiết lập ảnh–văn bản tổng hợp |

## 2. Nguồn gốc và phạm vi

BTXRD là viết tắt của *Bone Tumor X-ray Radiograph dataset*, được Yao và cộng sự công bố trong bài “A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors”, *Scientific Data*, tập 12, bài 88, năm 2025. Bộ dữ liệu hỗ trợ nghiên cứu phân lớp, định vị và phân vùng u xương nguyên phát, với ảnh tập trung ở chi trên, chi dưới và vùng chậu. [Bài báo gốc](https://www.nature.com/articles/s41597-024-04311-y).

Dữ liệu công khai được giới thiệu tại [trang dữ liệu BTXRD trên Figshare](https://doi.org/10.6084/m9.figshare.27865398). Trong XBone-Net, dữ liệu được tổ chức thành bài toán **phân lớp 10 lớp**; bộ nạp dữ liệu phân lớp không sử dụng chú thích vùng tổn thương làm nhãn huấn luyện.

## 3. Thành phần và đặc điểm dữ liệu

Phiên bản trong dự án gồm 3.746 ảnh, bảng siêu dữ liệu `dataset.csv` và 1.867 tệp chú thích `.json` trong `Annotations/`. Số liệu sau được tính từ bản dữ liệu cục bộ:

| Thuộc tính | Số lượng hoặc phạm vi |
| --- | ---: |
| Ảnh bình thường | 1.879 |
| Ảnh có u lành tính | 1.525 |
| Ảnh có u ác tính | 342 |
| Tổng ảnh có u | 1.867 |
| Mẫu có giới tính nam (`M`) | 2.098 |
| Mẫu có giới tính nữ (`F`) | 1.648 |
| Tuổi ghi trong siêu dữ liệu | 1–88 tuổi |
| Mẫu mang mã nguồn `center=1` | 2.938 |
| Mẫu mang mã nguồn `center=2` | 549 |
| Mẫu mang mã nguồn `center=3` | 259 |

Các thống kê tuổi và giới tính được tính theo bản ghi ảnh. Bảng hiện tại không có mã người bệnh ổn định để xác định số người bệnh duy nhất hoặc loại trừ khả năng một người có nhiều ảnh.

`dataset.csv` chứa mã ảnh (`image_id`), mã nguồn (`center`), tuổi (`age`), giới tính (`gender`), các cờ vị trí giải phẫu, vùng cơ thể, tư thế chụp (`frontal`, `lateral`, `oblique`), sự hiện diện của u (`tumor`), nhóm lành/ác tính (`benign`, `malignant`) và các cờ phân nhóm bệnh lý.

## 4. Hệ nhãn và phân bố lớp

Nhãn `normal` được suy ra khi `tumor=0`. Các lớp còn lại lấy từ các cờ bệnh lý. Thứ tự lớp dưới đây khớp cấu hình `configs/dataset/btxrd.yaml`; mã lớp bắt đầu từ 0. Tên tiếng Anh là chuỗi nhãn dùng trong mô hình, phần diễn giải tiếng Việt dùng để đọc tài liệu.

| Mã | Nhãn mô hình | Diễn giải tiếng Việt | Huấn luyện | Xác thực | Kiểm thử | Tổng |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | `normal` | Bình thường | 1.322 | 191 | 366 | 1.879 |
| 1 | `osteochondroma` | U xương sụn | 538 | 70 | 145 | 753 |
| 2 | `osteosarcoma` | Sarcoma xương | 200 | 32 | 65 | 297 |
| 3 | `multiple osteochondromas` | Đa u xương sụn | 189 | 29 | 45 | 263 |
| 4 | `simple bone cyst` | Nang xương đơn thuần | 149 | 17 | 40 | 206 |
| 5 | `other bt` | U lành tính khác | 83 | 8 | 24 | 115 |
| 6 | `giant cell tumor` | U tế bào khổng lồ | 54 | 8 | 31 | 93 |
| 7 | `synovial osteochondroma` | U sụn màng hoạt dịch | 35 | 7 | 9 | 51 |
| 8 | `other mt` | U ác tính khác | 27 | 5 | 13 | 45 |
| 9 | `osteofibroma` | U xơ xương | 24 | 8 | 12 | 44 |
| | **Tổng cộng** | | **2.621** | **375** | **750** | **3.746** |

Trong `btxrd-labels.csv` hiện tại, mỗi dòng có đúng một cờ lớp bằng 1. Bộ nạp cũng có quy tắc tương thích với nhãn phân cấp: nếu lớp con `multiple osteochondromas` hoặc `synovial osteochondroma` cùng bật với `osteochondroma`, lớp con được ưu tiên khi xác định nhãn đơn. Khi thống kê một phiên bản dữ liệu khác, cần áp dụng cùng quy tắc này để tránh đếm trùng.

Dữ liệu mất cân bằng rõ rệt: lớp `normal` chiếm khoảng **50,16%**, trong khi `osteofibroma` có 44 mẫu, khoảng **1,17%**. Số mẫu của lớp lớn nhất gấp khoảng **42,7 lần** lớp nhỏ nhất.

## 5. Cách chia tập

Quy trình trong `preprocess_btxrd.py` chia ngẫu nhiên ở **mức ảnh**, tỷ lệ mục tiêu 70/10/20. Mã nguồn không đặt tham số phân tầng theo lớp. Tập kiểm thử được tách trước, sau đó tập xác thực được tách từ phần còn lại.

| Tập | Số ảnh | Tỷ lệ thực tế |
| --- | ---: | ---: |
| Huấn luyện (`train`) | 2.621 | 69,97% |
| Xác thực (`validate`) | 375 | 10,01% |
| Kiểm thử (`test`) | 750 | 20,02% |
| **Tổng** | **3.746** | **100,00%** |

Các tập không trùng `image_id`. Tuy nhiên, do thiếu mã người bệnh ổn định, phép kiểm tra này chưa chứng minh ba tập độc lập ở mức người bệnh. Phép chia cũng không giữ riêng một nguồn `center` làm tập kiểm thử ngoài miền.

## 6. Văn bản tổng hợp

BTXRD trong dự án không có bệnh sử tự nhiên đi kèm từng ảnh. Văn bản tiếng Anh được tạo từ siêu dữ liệu như tuổi, giới tính, vị trí giải phẫu và tư thế chụp, kết hợp các mẫu câu về lý do vào viện, diễn tiến bệnh, khám thực thể và tiền sử.

Mã `generate_clinical_reports.py` chọn mẫu câu theo cả cờ `tumor` và `malignant`. Vì vậy, triệu chứng hoặc tiền sử được sinh không phải thông tin quan sát từ người bệnh; việc không viết trực tiếp tên lớp cũng không loại bỏ sự phụ thuộc giữa văn bản và nhãn. Kết quả dùng văn bản này không đủ để kết luận về lợi ích của bệnh sử thực tế trong môi trường lâm sàng.

**Cấu trúc đang có:** cả 3.746 tệp văn bản nằm trực tiếp tại `data/BTXRD/reports/<tên_ảnh_không_đuôi>.txt`. Các thư mục `reports/clinical/`, `reports/clinical_v2/` và `reports/xray/` chưa có trong bản dữ liệu kiểm tra. Bộ nạp `BTXRDDataset` có nhánh đọc văn bản trực tiếp từ `reports/` khi thiếu cấu trúc hai loại báo cáo.

Script sinh văn bản hiện ghi đầu ra vào `reports/clinical_v2/`. Đây là đường dẫn của một lần sinh mới, không phải vị trí văn bản đang dùng trong bản dữ liệu này; chưa xác minh toàn bộ văn bản hiện có được tạo bằng đúng phiên bản script hiện tại.

## 7. Tổ chức dữ liệu và quy trình sử dụng

### 7.1. Các tệp cần nhận biết

Các đường dẫn dưới đây tính từ thư mục gốc dự án.

| Đường dẫn | Nội dung |
| --- | --- |
| `data/BTXRD/images/` | Ảnh X-quang BTXRD |
| `data/BTXRD/Annotations/` | Chú thích vùng tổn thương `.json` |
| `data/BTXRD/dataset.csv` | Siêu dữ liệu ảnh BTXRD |
| `data/BTXRD/btxrd-labels.csv` | Mã ảnh và 10 cột nhãn |
| `data/BTXRD/btxrd-split.csv` | Mã ảnh và tên tập |
| `data/BTXRD/reports/` | Văn bản tổng hợp `.txt` hiện có, lưu trực tiếp trong thư mục này |

Ảnh và văn bản được ghép bằng tên cơ sở: bỏ phần mở rộng `.jpeg` của `image_id` rồi thêm `.txt`.

### 7.2. Các bước chuẩn bị đầu vào

1. **Chọn đúng phiên bản dữ liệu.** Dùng bảng nhãn, bảng chia tập, ảnh và văn bản thuộc cùng phiên bản; lưu lại phiên bản trước khi thay đổi dữ liệu.
2. **Ghép bảng theo `image_id`.** Bảng nhãn và bảng chia phải khớp một–một. Giá trị tên tập là `train`, `validate`, `test`; bộ nạp hỗ trợ ánh xạ `val` sang `validate`.
3. **Xác định nhãn.** Suy mã lớp từ 10 cờ nhãn theo thứ tự trong `configs/dataset/btxrd.yaml`.
4. **Ghép ảnh và văn bản.** Dùng các tệp văn bản tiếng Anh hiện có trong `reports/`. Văn bản được tổng hợp từ siêu dữ liệu có liên quan đến nhãn, cần giữ rõ đặc điểm này khi diễn giải kết quả.
5. **Tiền xử lý ảnh.** Bộ nạp chuyển ảnh sang RGB. Chiến lược `direct_resize` hiện chuyển ảnh gốc cho quy trình biến đổi của backbone; với XBone-Net/BiomedCLIP là resize, cắt vùng trung tâm và chuẩn hóa của backbone, không thêm bước đệm viền riêng.
6. **Chuẩn bị văn bản.** Bộ nạp đọc UTF-8, bỏ khoảng trắng hai đầu, chuyển chữ thường và dùng tokenizer của backbone. Độ dài token và biến đổi ảnh cụ thể cần đối chiếu cấu hình mô hình của thí nghiệm.
7. **Giữ đúng phạm vi đánh giá.** Huấn luyện trên `train`, chọn mô hình/thiết lập trên `validate`, báo cáo phân lớp trên `test`.

### 7.3. Kết quả kiểm tra cho tài liệu

- BTXRD: 3.746 dòng nhãn khớp một–một với 3.746 dòng chia tập; mỗi dòng có đúng một nhãn; đủ 3.746 ảnh và 3.746 tệp văn bản trực tiếp trong `reports/`, không có tệp rỗng trong các tài nguyên này.
- Phạm vi kiểm tra là tính nhất quán của bảng, số lượng, sự tồn tại và kích thước khác 0 của tệp. Chưa giải mã lại toàn bộ ảnh, tìm ảnh trùng nội dung, thẩm định nhãn y khoa. Không có dữ liệu hay phép chia nào được thay đổi khi lập tài liệu.

## 8. Nguồn đối chiếu

1. Yao, S., Huang, Y., Wang, X. và cộng sự (2025). *A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors*. Scientific Data, 12, 88. [DOI bài báo](https://doi.org/10.1038/s41597-024-04311-y); [DOI bộ dữ liệu](https://doi.org/10.6084/m9.figshare.27865398).
2. Số liệu: `data/BTXRD/dataset.csv`, `data/BTXRD/btxrd-labels.csv`, `data/BTXRD/btxrd-split.csv` và các tài nguyên ảnh/văn bản tương ứng.
3. Quy trình chuẩn bị: `data/BTXRD/preprocess_btxrd.py`, `data/BTXRD/generate_clinical_reports.py`.
4. Quy ước nạp dữ liệu: `src/datasets/btxrd.py`, `src/datasets/preprocessing.py`, `configs/dataset/btxrd.yaml`.
