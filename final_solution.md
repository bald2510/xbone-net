Sau các chỉnh sửa, XBone-Net hiện là một mô hình đa phương thức hai pha, kết hợp ảnh X-quang độ phân giải cao và báo cáo lâm sàng. Kiến trúc cuối cùng đã loại bỏ gate, learned skip, việc nối lặp raw global embeddings và prototypical head học được.

## 1. Pipeline tổng thể

```text
Ảnh X-quang
 ├─ Global image 224×224
 └─ Các local tile 224×224 + tọa độ
          ↓
BiomedCLIP Vision Encoder + LoRA
          ↓
Global visual token + local patch tokens
          ↓
Spatial Token Resampler
          ↓
25 visual tokens: 1 global + 24 resampled

Báo cáo lâm sàng
          ↓
PubMedBERT + LoRA
          ↓
Global text token + local text tokens

Visual tokens + Text tokens
          ↓
Cross-attention đối xứng global-to-local
          ↓
Image-enhanced + Text-enhanced
          ↓
Concatenate [hI ; hT]
          ↓
MLP Fusion 1024 → 512 → 512
          ↓
Empirical Centroid Classifier
          ↓
Cosine logits → Cross-Entropy
          ↓
Dự đoán lớp
```

OOD detection và explanation là các mô-đun hậu xử lý tùy chọn, không phải thành phần bắt buộc của forward classification.

---

## 2. Xử lý ảnh độ phân giải cao

Implementation: [high_resolution.py](C:/Users/lebat/Documents/Github/xbone-net/src/datasets/high_resolution.py)

Ảnh được biểu diễn bằng:

- Một global image được letterbox về hình vuông.
- Tập local tiles kích thước `224 × 224`.
- Tọa độ chuẩn hóa `[x1,y1,x2,y2]` cho từng tile.
- Mask để phân biệt tile thật và padding.
- Giới hạn tối đa 64 tiles.

### Điểm mạnh

- Giữ được cả ngữ cảnh toàn ảnh và chi tiết cục bộ.
- Chia tile có tính xác định, tái lập được.
- Bao phủ toàn bộ ảnh, không phụ thuộc detector hay ROI được gán nhãn.
- Tọa độ giúp mô hình phân biệt hai vùng có nội dung tương tự nhưng vị trí khác nhau.
- Dùng chung pipeline cho các nguồn ảnh khác nhau.

### Điểm yếu

- Tile không chồng lấn nên cấu trúc nằm trên ranh giới tile có thể bị chia cắt.
- Khi số tile vượt giới hạn, ảnh phải được giảm kích thước trước khi chia, làm mất một phần lợi ích high-resolution.
- Bộ lọc dựa trên độ lệch chuẩn có thể loại bỏ vùng đồng nhất nhưng vẫn có ý nghĩa.
- Đây là tiling hình học, không thích nghi theo nội dung hay giải phẫu.
- Chi phí tính toán tăng gần tuyến tính theo số tile.

---

## 3. BiomedCLIP backbone

Implementation: [biomedclip.py](C:/Users/lebat/Documents/Github/xbone-net/src/models/backbone/biomedclip.py)

Backbone bao gồm:

- ViT-B/16 cho ảnh.
- PubMedBERT cho báo cáo.
- Không gian embedding chung hình ảnh–văn bản.
- Trọng số pretrained gốc được đóng băng.
- LoRA được gắn vào cả image encoder và text encoder.

### Điểm mạnh

- Có prior ngữ nghĩa y sinh tốt hơn backbone thị giác thuần túy.
- Hai modality đã được ánh xạ vào không gian tương đối tương thích.
- ViT cung cấp patch tokens cần thiết cho xử lý ảnh cục bộ.
- PubMedBERT xử lý được phụ thuộc ngữ cảnh giữa các thuật ngữ trong báo cáo.

### Điểm yếu

- Global vision encoder vẫn nhận đầu vào 224×224.
- Sự tương thích pretrained chủ yếu ở embedding cấp toàn cục, không đảm bảo patch token và text token đã căn chỉnh tốt.
- Backbone lớn, đặc biệt tốn tài nguyên khi xử lý nhiều local tiles.
- Chất lượng biểu diễn phụ thuộc đáng kể vào prior mà BiomedCLIP đã học.

---

## 4. LoRA adaptation

LoRA được áp dụng lên attention và MLP của vision encoder, đồng thời lên các projection quan trọng của text encoder.

Local tile encoder hiện cho phép gradient đi vào visual LoRA; activation checkpointing và chunked encoding được dùng để kiểm soát bộ nhớ.

### Điểm mạnh

- Điều chỉnh được cả hai modality mà không cần cập nhật toàn bộ backbone.
- Giảm số tham số học được và dung lượng checkpoint.
- Giữ lại phần lớn tri thức pretrained.
- Local tiles thực sự tham gia quá trình thích nghi thay vì chỉ là đặc trưng đóng băng.

### Điểm yếu

- Low-rank update giới hạn khả năng thay đổi representation.
- Rank và vị trí gắn LoRA là các lựa chọn thủ công.
- Dù backbone bị đóng băng, LoRA vẫn có thể làm lệch không gian đa phương thức pretrained.
- Backpropagation qua tất cả tiles vẫn tốn thời gian; checkpointing tiết kiệm bộ nhớ bằng cách đổi lấy compute.

---

## 5. Local visual token pooling

Mỗi tile ban đầu có khoảng `14 × 14 = 196` patch tokens. Các patch này được adaptive-average-pool về lưới `2 × 2`, tức bốn token cho mỗi tile.

### Điểm mạnh

- Giảm mạnh số token đưa vào resampler.
- Vẫn giữ được bố cục không gian thô trong tile.
- Làm attention khả thi khi ảnh có nhiều tiles.

### Điểm yếu

- Nén 196 token xuống 4 token là một bottleneck mạnh.
- Tổn thương nhỏ có thể bị trung bình hóa.
- Pooling cố định không ưu tiên token quan trọng.
- Thông tin đã mất tại bước này không thể được resampler khôi phục.

---

## 6. Spatial Token Resampler

Resampler nhận:

- Global visual embedding.
- Local visual tokens.
- Tọa độ tile.
- Tile mask.

Sau đó sử dụng 24 learned queries qua hai lớp cross-attention để tạo 24 visual tokens cố định. Kết quả cuối cùng là:

\[
V=[g_I;r_1,\ldots,r_{24}]
\]

### Điểm mạnh

- Chuyển số tile biến thiên thành số token cố định.
- Cho phép mô hình tổng hợp mềm từ mọi vùng hợp lệ.
- Tọa độ cung cấp nhận thức không gian.
- Query được điều kiện hóa bởi global image nên việc lấy thông tin cục bộ có ngữ cảnh.
- Attention có thể ánh xạ ngược về các tile để trực quan hóa.

### Điểm yếu

- 24 queries tạo thêm một bottleneck thông tin.
- Các learned queries có thể học biểu diễn trùng lặp hoặc bị collapse.
- Số query cố định không thích nghi với độ phức tạp của từng ảnh.
- Attention weight không đồng nghĩa với mức độ đóng góp nhân quả.
- Resampler phải nén lượng thông tin rất lớn trước khi fusion nhìn thấy text.
- Phần nén visual diễn ra độc lập với report; report không thể yêu cầu resampler giữ lại một chi tiết đã bị loại bỏ.

---

## 7. Text token representation

Text encoder tạo:

- Global text representation.
- Chuỗi contextual local text tokens.
- Attention mask cho padding.

### Điểm mạnh

- Giữ được thông tin ở cấp từ/cụm từ, không chỉ sử dụng một vector report duy nhất.
- Cho phép global image truy vấn trực tiếp các phát hiện được mô tả trong report.
- Mask ngăn attention vào padding.

### Điểm yếu

- Report dài vẫn có thể bị truncation.
- Không có cơ chế riêng cho report rỗng, report sai hoặc report không liên quan.
- Không có token biểu diễn “không có thông tin đáng tin cậy”.
- Mô hình không trực tiếp ước lượng độ tin cậy của toàn report.

---

## 8. Phase 1: Semantic matching

Loss nằm trong [losses.py](C:/Users/lebat/Documents/Github/xbone-net/src/utils/losses.py).

Phase 1 căn chỉnh image và report embeddings bằng soft-target semantic matching:

\[
\mathcal L_{P1}
=
\mathcal L_{\text{semantic-matching}}(z_I,z_T,y).
\]

Cặp đúng có target similarity cao nhất; các mẫu cùng lớp có target similarity gần nhau thay vì bị xem hoàn toàn là negative.

### Điểm mạnh

- Tạo không gian đa phương thức có cấu trúc trước khi học classifier.
- Tránh coi mọi cặp ngoài đường chéo là negative tuyệt đối.
- Phù hợp với classifier dựa trên khoảng cách ở Phase 2.
- Giảm xung đột giữa việc căn chỉnh modality và việc tối ưu ranh giới phân lớp.

### Điểm yếu

- Target similarity là siêu tham số thủ công.
- Các mẫu cùng lớp bị kéo gần nhau dù có thể chứa các yếu tố ngữ nghĩa khác nhau.
- Loss phụ thuộc vào thành phần của mini-batch.
- Fusion module chưa được huấn luyện trực tiếp ở Phase 1.
- Mục tiêu alignment không hoàn toàn giống mục tiêu cosine-centroid ở Phase 2.

---

## 9. Fusion không sử dụng gate

Implementation: [cross_attention.py](C:/Users/lebat/Documents/Github/xbone-net/src/models/fusion/cross_attention.py)

Đầu tiên, visual và text tokens được pre-normalize và project về 512 chiều.

Hai hướng cross-attention là:

\[
c_{I\leftarrow T}
=
\operatorname{CrossAttn}(g_I,T_{\text{local}},T_{\text{local}})
\]

\[
c_{T\leftarrow I}
=
\operatorname{CrossAttn}(g_T,V_{\text{local}},V_{\text{local}})
\]

Residual trong từng nhánh:

\[
h_I=\operatorname{LN}(g_I+c_{I\leftarrow T})
\]

\[
h_T=\operatorname{LN}(g_T+c_{T\leftarrow I})
\]

Sau đó:

\[
z_F
=
\operatorname{MLP}\left([h_I;h_T]\right),
\qquad z_F\in\mathbb R^{512}.
\]

Không còn:

- Gated fusion.
- Learned skip.
- Nối lại raw global embeddings.
- Residual đối xứng sau MLP.

### Điểm mạnh

- Hai modality đều có thể truy vấn thông tin cục bộ của modality còn lại.
- Residual bảo toàn thông tin global trong từng nhánh.
- Pre-normalization giảm vấn đề chênh lệch scale.
- Không nối raw globals lần thứ hai nên tránh shortcut và dư thừa đặc trưng.
- Kiến trúc compact hơn, MLP chỉ nhận 1024 chiều.
- Mask được áp dụng đúng cho text padding và visual padding.
- Hai nhánh có ý nghĩa tương đối rõ ràng, thuận lợi cho ablation và explanation.

### Điểm yếu

- Đây là global-to-local attention, chưa phải tương tác đầy đủ giữa mọi visual và text token.
- Mỗi hướng chỉ sử dụng một global query, tạo bottleneck trước khi fusion.
- Không có gate nên mô hình không thể chủ động đưa đóng góp của report về gần 0.
- Softmax attention luôn phải phân bố xác suất lên một số token hợp lệ, kể cả khi toàn report không hữu ích.
- Concatenation mặc định cả hai nhánh luôn tồn tại và có giá trị.
- Không có learned skip quanh MLP nên toàn bộ embedding cuối cùng phụ thuộc vào MLP fusion.
- Attention cao chỉ biểu thị sự tương thích truy vấn–khóa, không chứng minh token đó đáng tin cậy.

Do đó, fusion hiện tại có thể học “report hữu ích cho dự đoán”, nhưng chưa có cơ chế tường minh để kết luận “report này không đáng tin”.

---

## 10. Empirical Centroid Classifier

Implementation: [empirical_centroid.py](C:/Users/lebat/Documents/Github/xbone-net/src/models/classifier/empirical_centroid.py) và [centroids.py](C:/Users/lebat/Documents/Github/xbone-net/src/utils/centroids.py).

Với lớp \(k\), centroid được tính trực tiếp từ fused embeddings:

\[
c_k
=
\frac{1}{N_k}\sum_{i:y_i=k}z_i.
\]

Logit:

\[
\ell_k
=
s\,
\frac{z_F^\top c_k}
{\lVert z_F\rVert_2\lVert c_k\rVert_2},
\qquad s=15.
\]

Centroid không phải tham số được optimizer cập nhật. Chúng được tính trước Phase 2, cập nhật sau mỗi epoch và tính lại khi xuất best model.

### Điểm mạnh

- Là prototype classifier đúng theo nghĩa empirical centroid.
- Không có trainable class weights nên classifier dễ diễn giải.
- Không tồn tại xung đột giữa prototype học bằng gradient và empirical centroid.
- Cosine classifier phù hợp với không gian embedding đã normalize.
- Có thể quan sát trực tiếp khoảng cách từ mẫu đến từng lớp.
- Dễ mở rộng lớp về mặt thuật toán: tính thêm centroid thay vì huấn luyện lại một linear head hoàn chỉnh.

### Điểm yếu

- Một centroid duy nhất giả định mỗi lớp có một cụm tương đối đơn mode.
- Không mô hình hóa được nhiều sub-cluster trong cùng lớp.
- Không xét covariance hay hình dạng phân bố của từng lớp.
- Nhạy với embedding ngoại lệ vì sử dụng phép trung bình.
- Bắt buộc mỗi lớp phải có ít nhất một mẫu khi tính centroid.
- Centroid cố định trong một epoch nên bị “stale” khi encoder đang thay đổi.
- Scale \(s=15\) là cố định thay vì được hiệu chỉnh hoặc học.
- Phải quản lý chặt việc tính lại centroid tương ứng với đúng checkpoint encoder.

---

## 11. Phase 2 loss

Phase 2 sử dụng weighted cross-entropy trên cosine logits:

\[
\mathcal L_{P2}
=
-\sum_k w_k y_k
\log
\frac{\exp(\ell_k)}
{\sum_j\exp(\ell_j)}.
\]

Gradient cập nhật:

- LoRA.
- Spatial resampler.
- Fusion module.

Gradient không cập nhật centroid.

### Điểm mạnh

- Loss đơn giản và nhất quán với đầu ra classifier.
- Không cần pull–push prototype loss riêng.
- Tránh nhiều hệ số margin và regularization khó cân bằng.
- Trọng số effective-number có thể kiểm soát đóng góp tương đối của các lớp.

### Điểm yếu

- Cross-entropy chỉ tối ưu xác suất tương đối, không trực tiếp ép mỗi lớp tạo cụm compact.
- Không trực tiếp tối ưu khoảng cách nội lớp hay separation tuyệt đối giữa centroids.
- Class weighting có thể làm gradient của một số lớp quá lớn.
- Việc xen kẽ cập nhật encoder bằng gradient và centroid bằng phép trung bình giống một quy trình gần-EM, nhưng không phải tối ưu đồng thời chính xác.
- Centroid được tính ở chế độ evaluation trong khi fusion được học với dropout, tạo ra sai khác nhỏ giữa representation dùng để học và representation dùng để tính tâm.

---

## 12. OOD detection tùy chọn

OOD sử dụng fused embedding, class centers và khoảng cách Mahalanobis với covariance dùng chung.

### Điểm mạnh

- Không cần thay đổi classifier chính.
- Có ý nghĩa hình học rõ ràng.
- Xét cả khoảng cách đến tâm và hướng biến thiên của embedding.

### Điểm yếu

- Giả định các lớp có cấu trúc gần Gaussian và chia sẻ covariance.
- Nghịch đảo covariance trong không gian cao chiều có thể không ổn định.
- Ngưỡng OOD là hậu kiểm, không được tối ưu cùng mô hình.
- Không đảm bảo confidence được calibration dưới mọi kiểu distribution shift.
- Trong cấu hình proposed chuẩn, `run_ood` hiện đang tắt nên đây chưa phải thành phần bắt buộc của XBone-Net.

---

## 13. Attention visualization và explanation

Hệ thống có thể trực quan hóa:

- Attention từ visual queries đến local tiles.
- Image-global đến text tokens.
- Text-global đến visual tokens.
- Gradient/Integrated Gradients.
- Branch ablation trong giai đoạn giải thích.

### Điểm mạnh

- Attention mask được giữ đúng khi tổng hợp trọng số.
- Có thể ánh xạ attention về tọa độ tile trên ảnh gốc.
- Hai nhánh fusion cho phép phân tích đóng góp ảnh và report riêng.
- Hữu ích để phát hiện shortcut hoặc sự không nhất quán giữa hai modality.

### Điểm yếu

- Attention không phải bằng chứng nhân quả.
- Branch masking tạo đầu vào ngoài phân bố mà mô hình gặp khi huấn luyện.
- Explanation chịu ảnh hưởng của các bottleneck pooling và resampler.
- Integrated Gradients và rollout làm tăng đáng kể thời gian suy luận.

---

## Đánh giá tổng thể

Điểm mạnh cốt lõi của kiến trúc là tính nhất quán:

\[
\text{multimodal alignment}
\rightarrow
\text{compact symmetric fusion}
\rightarrow
\text{cosine embedding}
\rightarrow
\text{empirical centroid}.
\]

Nó tránh được nhiều mâu thuẫn của phiên bản cũ: không còn prototype vừa được học bằng gradient vừa bị ghi đè bằng empirical centroid; không còn nối lặp raw global embeddings; không còn gate thiếu supervision; normalization và mask đã rõ ràng hơn.

Các điểm yếu quan trọng nhất còn lại là:

1. Có nhiều bottleneck liên tiếp: `196 → 4` token mỗi tile, toàn ảnh → 24 resampled tokens, mỗi modality → một enhanced global vector, và mỗi lớp → một centroid.
2. Local tile encoding và centroid recomputation có chi phí tính toán cao.
3. Fusion chưa có cơ chế xử lý modality thiếu, report sai hoặc report không đáng tin.
4. Phase 1 chưa huấn luyện trực tiếp fusion, tạo độ lệch giữa hai mục tiêu huấn luyện.
5. Một centroid cho mỗi lớp là giả định hình học khá mạnh.
6. Explanation dựa trên attention chỉ nên được xem là bằng chứng hỗ trợ, không phải giải thích nhân quả.

Vì vậy, phiên bản hiện tại có kiến trúc gọn, dễ diễn giải và phù hợp để làm proposed method. Tuy nhiên, tuyên bố khoa học nên giới hạn ở “multimodal high-resolution representation với empirical centroid classification”, chưa nên tuyên bố mô hình có khả năng đánh giá độ tin cậy của report hoặc phát hiện OOD như một năng lực nội tại.