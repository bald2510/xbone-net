# Mô tả bộ dữ liệu CTCH

## 1. Tổng quan

| Đặc điểm | Mô tả |
| --- | --- |
| Nguồn dữ liệu | Dữ liệu tại Bệnh viện Chấn thương Chỉnh hình Thành phố Hồ Chí Minh |
| Dữ liệu ảnh | Ảnh X-quang xương được lựa chọn, định dạng `.jpg` |
| Số mẫu phân lớp | 3.249 |
| Số lớp đích | 22 |
| Bài toán trong dự án | Phân lớp đa lớp, mỗi ảnh có một nhãn đích |
| Văn bản đi kèm | Bệnh sử tiếng Việt từ hồ sơ và bản dịch tiếng Anh |
| Huấn luyện / xác thực / kiểm thử | 2.265 / 315 / 669 |
| Đơn vị chia tập | Người bệnh, theo `patient_key` |
| Tập ngoài phân phối riêng | 18 mẫu trong `ctch-ood.csv`, nằm ngoài 3.249 mẫu phân lớp |
| Vai trò nghiên cứu | Khảo sát phân loại bệnh lý xương với bệnh sử thực tế và cảnh báo ngoài phân phối |

## 2. Nguồn gốc và phạm vi

Theo Chương 4 của báo cáo dự án, CTCH được thu thập tại Bệnh viện Chấn thương Chỉnh hình Thành phố Hồ Chí Minh, do giảng viên Võ Thế Hào và giảng viên Đỗ Phúc Thịnh cung cấp cho nghiên cứu. Nguồn dữ liệu gồm ảnh được chọn từ ca khám và các trường bệnh án tiếng Việt. Đây là thông tin nguồn gốc được ghi trong tài liệu nội bộ, không phải một bộ dữ liệu công khai có DOI đã được xác minh.

Phạm vi phân lớp sử dụng ảnh X-quang đại diện cho vị trí bệnh lý. Mã tiền xử lý loại các dòng đánh dấu xóa (`Deleted`), có dụng cụ cấy ghép (`Implants`) hoặc chưa chọn ảnh (`SelectedImage` rỗng), rồi tách nhóm OOD theo cờ dữ liệu nguồn. Thống kê trong tài liệu này lấy từ các bảng dữ liệu đã xuất, không suy số mẫu cuối cùng từ toàn bộ thư mục ảnh hoặc workbook nguồn.

## 3. Thành phần và văn bản đầu vào

Mỗi mẫu phân lớp gồm một ảnh, nhãn đơn và bệnh sử tương ứng. Văn bản lâm sàng được ghép từ ba trường:

| Trường nguồn | Ý nghĩa | Tiêu đề trong văn bản đầu vào |
| --- | --- | --- |
| `LyDoVaoVien_BenhAN` | Lý do vào viện | `Reason for admission` |
| `QuaTrinhBenhLi` | Quá trình/diễn tiến bệnh lý | `Disease history` |
| `TienSuBenh_BanThan_LucVaoVien` | Tiền sử bệnh bản thân lúc vào viện | `Personal medical history` |

Bản tiếng Việt nằm trong `reports/clinical_vi/`; bản dịch tiếng Anh nằm trong `reports/clinical/`. Các trường chẩn đoán xác định, `Label` và kết quả khảo sát hình ảnh không được ghép vào chuỗi bệnh sử theo danh sách `CLINICAL_FIELDS` hiện tại. Hai thư mục `reports/xray/` và `reports/xray_vi/` tồn tại nhưng không có tệp tương ứng với các mẫu trong bảng dữ liệu đang kiểm tra; không xem chúng là nguồn văn bản sẵn có của bộ dữ liệu này.

Việc loại các cột chẩn đoán trực tiếp giúp hạn chế rò rỉ nhãn. Tuy nhiên, bệnh sử vẫn có thể đề cập chẩn đoán hoặc điều trị trước đó, và bản dịch có thể có sai lệch thuật ngữ. Chất lượng ngữ nghĩa của từng bệnh sử và bản dịch chưa được đánh giá lại trong lần lập tài liệu này.

## 4. Hệ nhãn và phân bố lớp

CTCH có **3.249 mẫu trong phân phối (ID)** và **22 lớp**. Mỗi dòng của `ctch-labels.csv` chứa `class_id`, `mapped_class` và 22 cột nhãn nhị phân. Thứ tự sau khớp `labels.txt` và `configs/dataset/ctch.yaml`; toàn bộ dòng hiện tại có nhãn one-hot phù hợp với `class_id`.

| Mã | Nhãn | Huấn luyện | Xác thực | Kiểm thử | Tổng |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Bình thường | 760 | 108 | 218 | 1.086 |
| 1 | Gãy xương quay | 209 | 29 | 61 | 299 |
| 2 | Gãy xương cánh tay | 147 | 21 | 42 | 210 |
| 3 | Gãy xương đùi | 106 | 15 | 31 | 152 |
| 4 | Gãy xương ngón tay | 102 | 14 | 31 | 147 |
| 5 | Gãy xương chày | 99 | 14 | 29 | 142 |
| 6 | Gãy xương cẳng tay | 99 | 14 | 29 | 142 |
| 7 | Gãy xương cổ chân | 87 | 12 | 26 | 125 |
| 8 | Gãy xương cẳng chân | 85 | 12 | 25 | 122 |
| 9 | Gãy xương vùng vai | 70 | 10 | 21 | 101 |
| 10 | Gãy xương bàn chân | 68 | 9 | 21 | 98 |
| 11 | Gãy cột sống thắt lưng | 53 | 7 | 16 | 76 |
| 12 | Gãy xương bàn tay | 53 | 7 | 16 | 76 |
| 13 | Gãy xương ngón chân | 52 | 7 | 16 | 75 |
| 14 | Viêm khớp | 47 | 6 | 15 | 68 |
| 15 | Gãy xương trụ | 46 | 6 | 14 | 66 |
| 16 | Cắt cụt ngón tay | 42 | 6 | 13 | 61 |
| 17 | Gãy xương bánh chè | 41 | 5 | 13 | 59 |
| 18 | Gãy xương gót | 36 | 5 | 11 | 52 |
| 19 | Gãy cột sống ngực | 28 | 4 | 8 | 40 |
| 20 | Gãy xương thuyền | 19 | 2 | 7 | 28 |
| 21 | Gãy xương vùng chậu | 16 | 2 | 6 | 24 |
| | **Tổng cộng** | **2.265** | **315** | **669** | **3.249** |

Lớp Bình thường chiếm khoảng **33,43%**. Lớp Gãy xương vùng chậu chỉ có 24 mẫu, khoảng **0,74%**; số mẫu lớp lớn nhất gấp **45,25 lần** lớp nhỏ nhất. Một số lớp chỉ có 2 mẫu xác thực và 6–7 mẫu kiểm thử, nên các độ đo riêng từng lớp có thể biến động mạnh.

## 5. Cách chia tập và kiểm soát trùng người bệnh

Quy trình chia nhóm các mẫu theo `patient_key`, xác định lớp chính của từng nhóm rồi phân bổ người bệnh theo lớp với tỷ lệ mục tiêu 70/10/20. Khi số người bệnh của một lớp cho phép, thuật toán giữ ít nhất một người bệnh ở mỗi tập. Tỷ lệ thực tế có thể lệch tỷ lệ mục tiêu do làm tròn và yêu cầu giữ các nhóm nguyên vẹn.

| Tập | Số ảnh | Số `patient_key` duy nhất | Tỷ lệ ảnh |
| --- | ---: | ---: | ---: |
| Huấn luyện (`train`) | 2.265 | 2.265 | 69,71% |
| Xác thực (`validate`) | 315 | 315 | 9,70% |
| Kiểm thử (`test`) | 669 | 669 | 20,59% |
| **Tổng** | **3.249** | **3.249** | **100,00%** |

Trong bản chia hiện tại, mỗi mã người bệnh có một ảnh, không có mã rỗng và **không có `patient_key` xuất hiện ở nhiều tập**.

## 6. Tập ngoài phân phối (OOD)

`ctch-ood.csv` hiện chứa **18 ảnh của 18 mã người bệnh**, tách khỏi 3.249 mẫu ID. Theo cột `Nhóm bệnh`, phân bố là:

| Nhóm trong bảng OOD | Số mẫu |
| --- | ---: |
| Chấn thương | 6 |
| Viêm nhiễm | 7 |
| U xương | 5 |
| **Tổng** | **18** |

Đây là nhóm mẫu được dành riêng để khảo sát bệnh lý ngoài không gian nhãn phân lớp. Tên nhóm rộng như “Chấn thương” không tự nó xác định OOD, vì các lớp ID cũng bao gồm chấn thương; tư cách OOD của từng mẫu dựa trên bảng đã tuyển chọn và nhãn nguồn.

Kiểm tra hiện tại không phát hiện trùng `image_id` hoặc mã người bệnh giữa ID và OOD. Cả 18 mẫu đều có ảnh, bệnh sử tiếng Việt và bản dịch tiếng Anh không rỗng. Tổng số mẫu trong hai bảng ID và OOD là **3.267**, nhưng **số mẫu của bài toán phân lớp 22 lớp vẫn là 3.249**; OOD không được tính thành lớp thứ 23.

Tập OOD nhỏ và cùng nguồn dữ liệu CTCH, nên kết quả chỉ phản ánh quần thể được chọn. Nó chưa đại diện cho mọi bệnh lý ngoài nhãn, bệnh viện khác hoặc thiết bị chụp khác. Số lượng 18 là số hiện tại trong bảng OOD; cần đếm lại nếu bảng này thay đổi.

## 7. Quyền truy cập và thông tin còn thiếu

Theo báo cáo dự án, CTCH được dùng cho nghiên cứu theo thỏa thuận bảo mật, đã loại thông tin định danh trực tiếp và không công khai dữ liệu gốc. Tài liệu này chỉ trình bày thống kê tổng hợp, không chép bệnh án hoặc mã người bệnh. Các bảng nội bộ vẫn có mã liên kết và cần được quản lý như dữ liệu nghiên cứu hạn chế truy cập.

Các bảng nhãn/chia tập được kiểm tra không cung cấp đầy đủ thời gian thu thập, phân bố thiết bị chụp, tuổi/giới tính CTCH hoặc hồ sơ thẩm định nhãn và phê duyệt đạo đức. Những nội dung này cần lấy từ hồ sơ nguồn khi lập một bản mô tả để công bố; tài liệu không tự suy diễn các thông tin đó.

## 8. Tổ chức dữ liệu và quy trình sử dụng

### 8.1. Các tệp cần nhận biết

Các đường dẫn dưới đây tính từ thư mục gốc dự án.

| Đường dẫn | Nội dung |
| --- | --- |
| `data/CTCH/images/` | Ảnh CTCH; chọn mẫu sử dụng theo bảng ID/OOD |
| `data/CTCH/labels.txt` | Danh sách 22 lớp có thứ tự |
| `data/CTCH/ctch-labels.csv` | Mã ảnh, mã lớp, tên lớp và nhãn one-hot |
| `data/CTCH/ctch-split.csv` | Mã ảnh, tên tập và `patient_key` |
| `data/CTCH/ctch-ood.csv` | Danh sách mẫu dành riêng cho OOD và thông tin nhóm |
| `data/CTCH/reports/clinical_vi/` | Bệnh sử tiếng Việt |
| `data/CTCH/reports/clinical/` | Bản dịch bệnh sử tiếng Anh |

Ảnh và văn bản được ghép bằng tên cơ sở: bỏ phần mở rộng `.jpg` của `image_id` rồi thêm `.txt`. `image_id` được tạo từ mã dòng nguồn và tên ảnh đã chọn; `patient_key` là trường riêng phục vụ chia tập, không được suy từ tiền tố tên ảnh.

### 8.2. Các bước chuẩn bị đầu vào

1. **Chọn đúng phiên bản dữ liệu.** Dùng bảng nhãn, bảng chia tập, ảnh và văn bản thuộc cùng phiên bản; lưu lại phiên bản trước khi thay đổi dữ liệu.
2. **Ghép bảng theo `image_id`.** Bảng nhãn và bảng chia phải khớp một–một. Giá trị tên tập là `train`, `validate`, `test`; bộ nạp hỗ trợ ánh xạ `val` sang `validate`.
3. **Xác định nhãn.** Dùng `class_id` trong khoảng 0–21, đối chiếu với `mapped_class` và `labels.txt`.
4. **Ghép ảnh và văn bản.** Dùng bệnh sử trong `reports/clinical/` cho đầu vào tiếng Anh; bản tiếng Việt tương ứng nằm trong `reports/clinical_vi/`. Không đưa các cột nhãn hoặc mã người bệnh vào văn bản mô hình.
5. **Tiền xử lý ảnh.** Bộ nạp chuyển ảnh sang RGB. Chiến lược `direct_resize` hiện chuyển ảnh gốc cho quy trình biến đổi của backbone; với XBone-Net/BiomedCLIP là resize, cắt vùng trung tâm và chuẩn hóa của backbone, không thêm bước đệm viền riêng.
6. **Chuẩn bị văn bản.** Bộ nạp đọc UTF-8, bỏ khoảng trắng hai đầu, chuyển chữ thường và dùng tokenizer của backbone. Độ dài token và biến đổi ảnh cụ thể cần đối chiếu cấu hình mô hình của thí nghiệm.
7. **Giữ đúng phạm vi đánh giá.** Huấn luyện trên `train`, chọn mô hình/thiết lập trên `validate`, báo cáo phân lớp trên `test`. CTCH OOD được quản lý riêng và không bổ sung vào tập huấn luyện phân lớp của giao thức này.

### 8.3. Kết quả kiểm tra cho tài liệu

- CTCH ID: 3.249 dòng nhãn khớp một–một với 3.249 dòng chia tập; mã lớp, tên lớp và one-hot thống nhất; đủ ảnh và văn bản Việt/Anh không rỗng; không trùng người bệnh giữa các tập.
- CTCH OOD: 18 mẫu có đủ ảnh và văn bản Việt/Anh không rỗng; không trùng mã ảnh hoặc người bệnh với ID.
- Phạm vi kiểm tra là tính nhất quán của bảng, số lượng, sự tồn tại và kích thước khác 0 của tệp. Chưa giải mã lại toàn bộ ảnh, tìm ảnh trùng nội dung, thẩm định nhãn y khoa hoặc đánh giá lại chất lượng dịch. Không có dữ liệu hay phép chia nào được thay đổi khi lập tài liệu.

## 9. Nguồn đối chiếu

1. Số liệu: `data/CTCH/ctch-labels.csv`, `data/CTCH/ctch-split.csv`, `data/CTCH/ctch-ood.csv`, `data/CTCH/labels.txt` và các tài nguyên tương ứng.
2. Quy trình chuẩn bị: `data/CTCH/preprocess_ctch.py`.
3. Quy ước nạp dữ liệu: `src/datasets/ctch.py`, `src/datasets/preprocessing.py`, `configs/dataset/ctch.yaml`.
4. Nguồn gốc và điều kiện sử dụng: `docs/report/Chapter4/chapter4.tex`, mục chuẩn bị dữ liệu.
