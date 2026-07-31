Đề xuất đổi tên thành **“Cơ sở lý thuyết và các công trình liên quan”**, vì chương không chỉ tổng quan nghiên cứu mà còn giải thích các phương pháp nền tảng.

# Chương 2. Cơ sở lý thuyết và các công trình liên quan

## 2.1. Tổng quan chương

- Nêu phạm vi khảo sát.
- Giới thiệu sáu nhóm vấn đề:
  - Dữ liệu ảnh–văn bản y khoa.
  - Dung hợp đa phương thức.
  - Ảnh độ phân giải cao.
  - Tinh chỉnh hiệu quả tham số.
  - Phân loại mất cân bằng và tâm lớp.
  - OOD và khả năng giải thích.
- Nêu rằng phần cuối chương sẽ tổng hợp khoảng trống nghiên cứu và cơ sở thiết kế XBone-Net.

## 2.2. Học sâu đa phương thức trong phân tích ảnh y khoa

### 2.2.1. Các nguồn văn bản trong hệ thống y khoa

Phân biệt rõ:

- Bệnh sử lâm sàng trước chẩn đoán.
- Báo cáo đọc ảnh X-quang.
- Chẩn đoán xác định.
- Văn bản tổng hợp từ siêu dữ liệu.

Trình bày nguy cơ rò rỉ nhãn và liên hệ:

- CTCH sử dụng bệnh sử thật.
- BTXRD không có bệnh sử nên sử dụng văn bản tổng hợp.
- Kết quả BTXRD không được diễn giải tương đương bệnh sử tự nhiên.

### 2.2.2. Các mô hình ảnh–văn bản xây dựng từ đầu

Trình bày ngắn gọn:

- MLP/CNN fusion.
- TieNet.
- IRENE.
- Ưu điểm và yêu cầu dữ liệu ghép cặp lớn.

### 2.2.3. Mô hình ngôn ngữ–thị giác y khoa

- CLIP và học tương phản.
- ConVIRT.
- GLoRIA.
- MedCLIP.
- PubMedCLIP.
- BioMedCLIP.
- Các mô hình đa phương thức sinh như LLaVA-Med chỉ nên được nhắc ngắn để đối chiếu.

### 2.2.4. Cơ sở lựa chọn BioMedCLIP

- Tiền huấn luyện trên dữ liệu y sinh.
- Có bộ mã hóa ảnh và văn bản phù hợp.
- Hỗ trợ chuyển giao sang tác vụ phân loại.
- Kết luận đây là backbone được khảo sát, không khẳng định trước rằng chắc chắn tốt nhất.

## 2.3. Các chiến lược dung hợp ảnh và văn bản

### 2.3.1. Dung hợp sớm, trung gian và muộn

Giải thích cấp độ mà hai phương thức tương tác.

### 2.3.2. Phép nối đặc trưng

- Nguyên lý.
- Chi phí thấp.
- Không mô hình hóa tương tác chi tiết giữa token ảnh và token văn bản.

### 2.3.3. Chú ý chéo một chiều

- Văn bản truy vấn ảnh.
- Ảnh truy vấn văn bản.
- Ưu điểm và hạn chế của từng hướng.

### 2.3.4. Chú ý chéo hai chiều

- Hai phương thức trao đổi thông tin.
- Khả năng mô hình hóa quan hệ token.
- Chi phí tính toán và nguy cơ dư thừa thông tin.

### 2.3.5. So sánh các chiến lược dung hợp

Dùng một bảng:

| Phương pháp | Mức tương tác | Chi phí | Hạn chế |
|---|---:|---:|---|
| Nối đặc trưng | Toàn cục | Thấp | Không tương tác token |
| Chú ý một chiều | Token | Trung bình | Một phương thức phụ thuộc hướng truy vấn |
| Chú ý hai chiều | Token hai phía | Cao | Có thể tăng chi phí và dư thừa |

Kết đoạn bằng cơ sở khảo sát cross-attention hai chiều và concat trong XBone-Net.

## 2.4. Biểu diễn ảnh X-quang độ phân giải cao

### 2.4.1. Hạn chế của phép co ảnh toàn cục

- Mất đường gãy hoặc tổn thương nhỏ.
- Biến dạng tỷ lệ ảnh.

### 2.4.2. Chia ảnh thành vùng và xử lý đa tỷ lệ

- Tiling.
- AnyRes/Higher-AnyRes.
- Global–local.
- Các tiếp cận y khoa liên quan.

### 2.4.3. Nén token bằng latent resampling

- Perceiver.
- Flamingo.
- Đánh đổi giữa số token và bảo toàn chi tiết.

### 2.4.4. Biểu diễn vị trí và tọa độ

Giải thích vì sao token vùng ảnh cần giữ thông tin vị trí trên ảnh nguồn.

### 2.4.5. Tổng hợp đánh đổi

Giữ bảng chiến lược hiện có, sau đó kết luận global–local với ngân sách token cố định là lựa chọn cần kiểm chứng ở Chương 4.

Không nên đưa các chi tiết như chính xác bốn vùng, trọng số chấm điểm hoặc thuật toán sparse-focal vào Chương 2; các nội dung đó thuộc Chương 3.

## 2.5. Thích nghi mô hình nền tảng

### 2.5.1. Tinh chỉnh toàn phần và dò tuyến tính

- Khả năng thích nghi.
- Chi phí huấn luyện.
- Nguy cơ quá khớp.

### 2.5.2. Tổng quan PEFT

Phân loại chính xác thành:

- Prompt tuning.
- Adapter tuning.
- Tái tham số hóa hạng thấp.
- Tinh chỉnh chọn lọc tham số gốc.

### 2.5.3. Prompt tuning

Giữ PLIP và BiomedCoOp làm ví dụ.

### 2.5.4. Adapter tuning

Giữ CLIPath và FTB.

### 2.5.5. LoRA và tái tham số hóa hạng thấp

- Công thức cập nhật hạng thấp ở mức khái quát.
- LoRA có thêm tham số huấn luyện.
- Có thể hợp nhất cập nhật vào trọng số nền.
- Giảm số tham số cần cập nhật nhưng không đảm bảo toàn hệ thống suy luận nhanh hơn.

### 2.5.6. Cơ sở lựa chọn LoRA

Liên hệ trực tiếp với:

- Dữ liệu hạn chế.
- Few-shot.
- So sánh LoRA với full fine-tuning.
- Độ trễ và bộ nhớ được đo riêng ở Chương 4.

## 2.6. Học biểu diễn và phân loại dữ liệu mất cân bằng

### 2.6.1. Học tương phản ảnh–văn bản

- InfoNCE.
- Âm tính giả giữa các mẫu cùng lớp.
- Semantic Matching Loss và phân phối đích mềm.

### 2.6.2. Vấn đề mất cân bằng lớp

- Accuracy có thể bị chi phối bởi lớp phổ biến.
- Ảnh hưởng đến ranh giới quyết định và tâm lớp.

### 2.6.3. Các phương pháp xử lý mất cân bằng

- Lấy mẫu lại.
- Cross-Entropy có trọng số.
- Số mẫu hiệu dụng.
- Ưu, nhược điểm.

### 2.6.4. Mạng nguyên mẫu

- ProtoNet tiêu chuẩn.
- Phân loại theo khoảng cách đến prototype.
- Ưu điểm trong dữ liệu ít mẫu.

### 2.6.5. Tâm lớp thực nghiệm

Phân biệt rõ với ProtoNet episodic:

- Tâm được ước lượng từ embedding của tập huấn luyện.
- Không phải tham số tự do nhận đạo hàm.
- Có thể kết hợp cosine similarity.
- Cung cấp không gian hình học cho phân loại và OOD.

## 2.7. Phát hiện dữ liệu ngoài phân phối

Chỉ giữ một section OOD duy nhất.

### 2.7.1. Khái niệm ID, OOD và các dạng dịch chuyển

- Semantic OOD gần phân phối.
- Dịch chuyển liên bộ dữ liệu.
- Không đồng nhất hai kịch bản.

### 2.7.2. Phương pháp dựa trên độ bất định đầu ra

- MSP, Energy và ODIN chỉ giới thiệu ngắn.
- Trình bày kỹ entropy dự đoán vì được dùng trong thực nghiệm.

### 2.7.3. Khoảng cách đến tâm lớp

- Euclidean.
- Cosine-centroid.

### 2.7.4. Khoảng cách Mahalanobis

- Vai trò của ma trận hiệp phương sai.
- Điều chuẩn ma trận.
- Hạn chế khi số mẫu ít hoặc embedding có số chiều cao.

### 2.7.5. Phát hiện dựa trên kNN

- Khoảng cách đến các láng giềng huấn luyện.
- Lựa chọn \(k\).
- Chi phí lưu trữ và truy vấn.

### 2.7.6. Tổng hợp phương pháp được đánh giá

Lập bảng đúng với Chương 4:

| Phương pháp | Nguồn điểm | Có dùng tâm lớp | Có dùng hiệp phương sai |
|---|---|---:|---:|
| Cosine-centroid | Embedding | Có | Không |
| Mahalanobis-centroid | Embedding | Có | Có |
| Cosine-kNN | Embedding huấn luyện | Không | Không |
| Entropy | Xác suất dự đoán | Không | Không |

Không khẳng định phương pháp sẽ “từ chối hiệu quả Near-OOD”; chỉ nói đây là giả thuyết cần kiểm chứng.

## 2.8. Giải thích quyết định của mô hình đa phương thức

### 2.8.1. Attention không đồng nghĩa với giải thích

Phân biệt attention weight và attribution.

### 2.8.2. Các phương pháp dựa trên gradient

- Saliency.
- Gradient × Input.
- Integrated Gradients.

### 2.8.3. Integrated Gradients

- Baseline.
- Tích phân theo đường từ baseline đến đầu vào.
- Attribution cho ảnh và token văn bản.

### 2.8.4. Đánh giá độ tin cậy của attribution

- Can thiệp thay thế đầu vào.
- Deletion/insertion.
- Hạn chế của phân tích định tính.

Kết luận rằng luận văn chỉ sử dụng IG để truy vết quyết định ở cấp độ trường hợp, không đánh giá định vị tổn thương.

## 2.9. Khoảng trống nghiên cứu và cơ sở thiết kế XBone-Net

Nên dùng bảng liên kết trực tiếp:

| Khoảng trống | Thành phần XBone-Net | Nội dung đánh giá |
|---|---|---|
| Thiếu khai thác bệnh sử | Nhánh văn bản và dung hợp đa phương thức | So sánh với mô hình chỉ dùng ảnh |
| Mất chi tiết khi co ảnh | Nhánh global–local | Ablation không high-resolution |
| Dữ liệu hạn chế | LoRA và huấn luyện hai giai đoạn | Few-shot, full-shot, hiệu quả tính toán |
| Mất cân bằng lớp | Tâm lớp và CE có trọng số | Macro-F1, BAcc, AUROC, AUPRC |
| Mẫu ngoài phân phối | Bốn điểm OOD hậu xử lý | Semantic OOD và cross-dataset OOD |
| Khó truy vết quyết định | Integrated Gradients | Phân tích trường hợp đúng và sai |

## 2.10. Tóm tắt chương

- Tóm tắt các hướng nghiên cứu.
- Nhắc lại khoảng trống.
- Dẫn sang Chương 3 mà không trình bày trước kết quả Chương 4.

Outline này tạo được luồng xuyên suốt:

```text
Nguồn dữ liệu
→ Mã hóa ảnh và văn bản
→ Dung hợp
→ Thích nghi tham số
→ Phân loại mất cân bằng
→ OOD
→ Giải thích
→ Khoảng trống và thiết kế XBone-Net
```