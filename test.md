Dựa trên các kết quả đã hoàn tất, thành phần có bằng chứng rõ nhất đang làm giảm hiệu quả phân loại là **Phase 1 semantic matching hiện tại**. Hai thành phần tiếp theo chưa tạo được lợi ích rõ ràng là nhánh sparse-focal high-resolution và cross-attention. Chưa có đủ bằng chứng để quy trách nhiệm trực tiếp cho empirical-centroid.

## 1. Bằng chứng chính trên CTCH

| Thiết lập | Accuracy | Balanced Acc. | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|---:|
| Proposed đầy đủ | 0.5830 | 0.6049 | 0.5646 | 0.9568 |
| **Bỏ Phase 1** | **0.6019** | **0.6140** | **0.5858** | **0.9593** |
| Letterbox một ảnh | 0.5954 | 0.6028 | 0.5781 | 0.9581 |
| Direct resize, không high-res | 0.5914 | 0.5783 | 0.5547 | 0.9500 |
| Concat fusion | **0.6343** | 0.5773 | 0.5679 | 0.9574 |
| LoRA-PubMedCLIP baseline | 0.6203 | 0.6091 | 0.5816 | 0.9275 |
| LoRA-BiomedCLIP baseline | 0.5959 | 0.6092 | 0.5599 | 0.9334 |

Khi bỏ Phase 1, macro-F1 tăng:

\[
0.5858-0.5646=0.0212.
\]

Macro-F1 của Phase-2-only cao hơn proposed ở cả ba seed:

| Seed | Proposed | Phase-2-only | Chênh lệch |
|---:|---:|---:|---:|
| 42 | 0.5723 | 0.5731 | +0.0008 |
| 123 | 0.5557 | 0.5996 | +0.0439 |
| 456 | 0.5658 | 0.5847 | +0.0189 |

Quan trọng hơn, Phase-2-only đạt macro-F1 0.5858, cao hơn LoRA-PubMedCLIP 0.5816. Như vậy, nếu mục tiêu là vượt baseline về phân loại, kiến trúc Phase 2 hiện tại có khả năng làm được; bước tiền căn chỉnh Phase 1 đang làm mất phần lợi thế đó.

Do mới có ba seed, đây vẫn là bằng chứng định hướng, chưa phải kiểm định nhân quả có độ mạnh thống kê cao.

# 2. Thành phần gây suy giảm chính: Phase 1 semantic matching

Vấn đề không nhất thiết chỉ nằm ở hệ số local 0.25 mà ở toàn bộ thiết kế Phase 1.

## 2.1. Negative transfer từ Phase 1

BiomedCLIP vốn đã được căn chỉnh ảnh–văn bản trên dữ liệu y sinh. Phase 1 tiếp tục tối ưu hóa trên một tập CTCH tương đối nhỏ bằng một mục tiêu khác với phân loại cuối cùng:

\[
\mathcal L_{\mathrm{P1}}=\mathcal L_{\mathrm{contrastive}},
\qquad
\mathcal L_{\mathrm{P2}}=\mathcal L_{\mathrm{classification}}.
\]

Tối ưu hóa tương đồng ảnh–bệnh sử không bảo đảm rằng embedding sẽ có biên quyết định tốt hơn cho 22 lớp giải phẫu. Phase 1 có thể kéo các mẫu cùng nhãn lại gần nhưng đồng thời làm mất các chi tiết hữu ích cho phân loại.

## 2.2. Soft target 0.95 quá mạnh

Với hai mẫu khác nhau nhưng cùng lớp:

\[
\widehat Y_{ij}=0.95.
\]

Điều này gần bằng mức của cặp ảnh–báo cáo thực:

\[
\widehat Y_{ii}=1.
\]

Đối với các lớp như “gãy xương quay” hoặc “gãy xương đùi”, các mẫu cùng lớp vẫn có thể khác nhau về:

- vị trí tổn thương;
- tư thế chụp;
- mức độ gãy;
- góc chụp;
- đặc điểm bệnh sử.

Ép các mẫu này có độ tương đồng mục tiêu gần 1 có thể làm giảm phương sai nội lớp quá mức và tạo representation kém linh hoạt.

## 2.3. Contrastive batch quá nhỏ

Phase 1 chỉ nhìn thấy ma trận:

\[
\mathbf S\in\mathbb R^{16\times16}.
\]

Do đó, mỗi ảnh chỉ có tối đa 15 mẫu đối chiếu trong một bước. Gradient accumulation không làm tăng số cặp contrastive thực tế. Với 22 lớp CTCH, nhiều batch có thể không chứa đủ các lớp hoặc đủ hard negatives.

## 2.4. Nhánh local nhận tín hiệu yếu

Phase 1 sử dụng:

\[
\mathbf v=
\operatorname{Norm}
\left(
\mathbf g_v+0.25\,\overline{\mathbf v}_l
\right).
\]

Mỗi local token nhận mức ảnh hưởng gần đúng:

\[
\frac{0.25}{8}=0.03125.
\]

Vì vậy Phase 1 chủ yếu điều chỉnh global representation. Trong khi đó, đóng góp được kỳ vọng của XBone-Net lại nằm ở thông tin tổn thương cục bộ. Điều này tạo ra sự không nhất quán giữa mục tiêu Phase 1 và thiết kế Phase 2.

### Kết luận cho Phase 1

Phase 1 hiện tại là thành phần có bằng chứng mạnh nhất gây suy giảm macro-F1. Tuy nhiên, cần thêm ablation để phân biệt nguyên nhân cụ thể giữa:

- soft target 0.95;
- hệ số local 0.25;
- phép mean pooling local;
- batch contrastive 16;
- negative transfer từ bệnh sử.

# 3. Nhánh high-resolution: hữu ích nhưng chưa đủ mạnh

So với direct resize:

\[
\Delta\mathrm{MacroF1}
=
0.5646-0.5547
=
+0.0099,
\]

\[
\Delta\mathrm{BalancedAcc}
=
0.6049-0.5783
=
+0.0266.
\]

Như vậy, high-resolution không phải thành phần gây suy giảm hoàn toàn. Nó cải thiện độ nhạy cân bằng và macro-F1 so với direct resize.

Tuy nhiên, letterbox một ảnh lại đạt:

\[
\mathrm{MacroF1}=0.5781>0.5646.
\]

Điều này cho thấy sparse-focal chưa chứng minh được lợi ích rõ ràng so với một preprocessing đơn giản hơn.

Các điểm hạn chế có thể nằm ở:

1. Mỗi tile bị nén từ 196 patch thành một pooled token.
2. Tile CLS và pooled token dùng chung bounding box.
3. Bốn tile được chọn bằng texture và entropy, không có tín hiệu định vị tổn thương.
4. Local token được mean-pool trong Phase 1.
5. Một số tile có thể chứa cấu trúc sắc nét nhưng không phải tổn thương.

Do đó, high-resolution hiện nên được xem là thành phần chưa khai thác hết tiềm năng, không phải nguyên nhân chính gây giảm hiệu năng.

# 4. Cross-attention: chưa tạo lợi ích phân loại rõ ràng

So sánh với concat:

| Fusion | Accuracy | Balanced Acc. | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| Bidirectional cross-attention | 0.5830 | **0.6049** | 0.5646 | 0.9568 |
| Concat | **0.6343** | 0.5773 | **0.5679** | **0.9574** |

Cross-attention:

- tăng Balanced Accuracy;
- giảm Accuracy khá rõ;
- không cải thiện macro-F1;
- gần như không thay đổi AUROC.

Điều đó cho thấy fusion hai chiều đang thay đổi sự cân bằng giữa các lớp nhưng chưa làm representation phân loại tốt hơn một cách nhất quán.

Dù vậy, Phase-2-only vẫn dùng cross-attention và đạt macro-F1 cao nhất. Vì thế cross-attention không phải nguyên nhân chính; nó chỉ chưa thể hiện ưu thế rõ so với concat.

# 5. Empirical centroid: đáng nghi nhưng chưa được xác nhận

BTXRD thể hiện một mẫu quan trọng:

| Mô hình | Accuracy | Balanced Acc. | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|---:|
| Proposed | **0.8324** | 0.5878 | 0.6001 | **0.9383** |
| LoRA-PubMedCLIP | 0.8316 | **0.6457** | **0.6397** | 0.9031 |
| LoRA-BiomedCLIP | 0.8302 | 0.6339 | 0.6326 | 0.8771 |

Proposed có ranking rất tốt, thể hiện qua AUROC cao, nhưng macro-F1 và Balanced Accuracy thấp hơn. Điều này thường gợi ý vấn đề nằm ở biên quyết định:

\[
\ell_c=15\cos(\mathbf z,\boldsymbol\mu_c).
\]

Một tâm duy nhất cho mỗi lớp có thể không phù hợp khi lớp chứa nhiều kiểu hình, vị trí giải phẫu hoặc dạng biểu hiện khác nhau.

Tuy nhiên, linear probe hậu nghiệm trên fused embedding CTCH đạt macro-F1 trung bình xấp xỉ empirical-centroid, không tạo cải thiện rõ ràng. Điều này cho thấy không thể kết luận rằng chỉ cần thay centroid bằng linear head là đủ.

Quan trọng hơn, hiện chưa có kết quả hoàn chỉnh cho:

- `architecture/classifier/linear`;
- `architecture/classifier/no_class_weight`.

Vì vậy empirical-centroid hiện là nghi vấn thứ hai, chưa phải nguyên nhân đã được chứng minh.

# 6. Những thành phần không phải nguyên nhân chính

## LoRA

| Thiết lập | Macro-F1 |
|---|---:|
| Proposed LoRA | **0.5646** |
| Không fine-tune | 0.5405 |
| Full fine-tuning | 0.5357 |

LoRA tốt hơn cả đóng băng hoàn toàn và full fine-tuning. Đây là thành phần có lợi.

## Dữ liệu đa phương thức

| Dữ liệu đầu vào | Macro-F1 |
|---|---:|
| Ảnh + bệnh sử | **0.5646** |
| Text-only | 0.4770 |
| Image-only | 0.3984 |
| Báo cáo bị xáo trộn | 0.3039 |

Hai phương thức bổ trợ thực sự cho nhau. Đặc biệt, kết quả shuffled-report giảm mạnh cho thấy mô hình sử dụng quan hệ đúng giữa ảnh và bệnh sử, không chỉ khai thác phân bố văn bản.

## Clinical report

Thay bệnh sử ở Phase 1 bằng báo cáo X-quang cho macro-F1 0.5586, thấp hơn proposed 0.5646. Vì vậy lựa chọn clinical report hiện hợp lý hơn.

# 7. Xếp hạng nguyên nhân theo bằng chứng hiện có

| Mức độ | Thành phần | Nhận định |
|---|---|---|
| **1 — bằng chứng mạnh nhất** | Phase 1 semantic matching | Phase-2-only tăng macro-F1 ở cả ba seed và vượt LoRA-PubMedCLIP trên CTCH |
| **2 — hạn chế thứ cấp** | Sparse-focal và tổng hợp local | Tốt hơn direct resize nhưng không tốt hơn rõ ràng so với letterbox |
| **3 — chưa tạo lợi ích rõ** | Bidirectional cross-attention | BAcc tăng nhưng Accuracy và macro-F1 không vượt concat |
| **4 — đáng nghi, chưa xác nhận** | Empirical centroid | AUROC cao nhưng macro-F1 thấp; chưa có classifier ablation hoàn chỉnh |
| Không phải nguyên nhân | LoRA | Tốt hơn no-FT và full-FT |
| Không phải nguyên nhân | Multimodal input | Vượt rõ image-only, text-only và shuffled-report |

## Kết luận

Nếu phải chọn một thành phần duy nhất, **Phase 1 semantic matching theo thiết kế hiện tại là thành phần đang ngăn pipeline đạt kết quả phân loại vượt trội trên CTCH**.

Vấn đề có thể là sự kết hợp của soft target 0.95, batch contrastive nhỏ, mean pooling local và hệ số local 0.25—không nên quy hoàn toàn cho riêng 0.25.

Đối với BTXRD, bằng chứng lại nghiêng về **bước chuyển fused representation thành nhãn**, vì proposed có AUROC cao nhất nhưng macro-F1 thấp hơn. Tuy nhiên cần hoàn thành hai ablation `linear` và `no_class_weight` trước khi kết luận empirical-centroid là nguyên nhân.

Theo bằng chứng hiện có, cấu hình tốt nhất nên bắt đầu từ **Phase-2-only**, sau đó lần lượt tối ưu đầu phân loại và cách xử lý high-resolution. Không nên sửa đồng thời tất cả thành phần vì sẽ không xác định được nguồn cải thiện.

# 1. Điều chỉnh ưu tiên cao nhất: bỏ Phase 1 hiện tại

Trên CTCH:

\[
\text{Macro-F1: }0.5646\rightarrow0.5858,
\]

khi bỏ Phase 1. Đây là cải thiện lớn nhất đã được quan sát trực tiếp.

Cấu hình nên dùng:

```yaml
params:
  run_phase1: false

  phase1:
    enabled: false

  phase2:
    enabled: true
    init_from_phase1_checkpoint: false
    init_from_phase1_merged: false
```

Có thể chạy trực tiếp cấu hình hiện có:

```powershell
python train.py +experiment=ctch/ablation_study/architecture/phase/phase2_only seed=42
```

Cần tạo và chạy cấu hình tương đương cho BTXRD trước khi thay đổi canonical model.

Nếu mục tiêu chính của luận văn là phân loại, Phase-2-only hiện là ứng viên canonical hợp lý hơn mô hình hai pha.

---

# 2. Hoàn thành ablation đầu phân loại

BTXRD cho thấy proposed có AUROC cao nhưng macro-F1 thấp:

- proposed: AUROC 0.9383, macro-F1 0.6001;
- LoRA-PubMedCLIP: AUROC 0.9031, macro-F1 0.6397.

Điều này cho thấy representation có khả năng xếp hạng tốt, nhưng biên quyết định chưa tối ưu.

Nên so sánh ba đầu phân loại trên cùng Phase-2-only embedding:

## 2.1. Empirical centroid hiện tại

\[
\ell_{ic}=15\cos(\mathbf z_i,\boldsymbol\mu_c).
\]

## 2.2. Linear head

\[
\boldsymbol\ell_i=W\mathbf z_i+\mathbf b.
\]

Cấu hình đề xuất:

```yaml
params:
  phase2:
    classifier_type: linear
    loss_type: ce
    loss:
      label_smoothing: 0.05
      use_class_weights: true
      weight_type: effective_num
      effective_num_beta: 0.999
```

Linear head phải được huấn luyện đồng thời với fusion và LoRA. Linear probe trên embedding đã huấn luyện cho centroid không đủ để kết luận, vì representation đó vốn đã được tối ưu theo centroid.

## 2.3. Cosine classifier học được

Một lựa chọn trung gian phù hợp hơn:

\[
\ell_{ic}
=
s\,
\frac{\mathbf z_i^\top\mathbf w_c}
{\|\mathbf z_i\|_2\|\mathbf w_c\|_2},
\]

trong đó \(\mathbf w_c\) và \(s\) đều học được. Cách này giữ hình học cosine nhưng không buộc vector đại diện lớp phải bằng trung bình thực nghiệm.

Thứ tự nên thử:

1. Linear head;
2. cosine classifier học được;
3. empirical centroid hiện tại.

---

# 3. Điều chỉnh class weighting và label smoothing

Proposed sử dụng:

```yaml
label_smoothing: 0.0
use_class_weights: true
effective_num_beta: 0.999
```

Trong khi baseline LoRA dùng label smoothing 0.1. Cần chạy tối thiểu:

| Thiết lập | Class weight | Label smoothing |
|---|---|---:|
| A | Không | 0.0 |
| B | Effective number | 0.0 |
| C | Effective number | 0.05 |
| D | Effective number | 0.10 |

Khuyến nghị ban đầu:

```yaml
loss:
  use_class_weights: true
  weight_type: effective_num
  effective_num_beta: 0.999
  label_smoothing: 0.05
```

Nếu class weighting làm tăng recall nhưng giảm precision quá nhiều, có thể giảm:

```yaml
effective_num_beta: 0.99
```

Không nên đồng thời dùng class weight rất mạnh và weighted sampler, vì có thể hiệu chỉnh mất cân bằng hai lần.

---

# 4. Giữ LoRA, không chuyển sang full fine-tuning

Kết quả CTCH:

| Cách tinh chỉnh | Macro-F1 |
|---|---:|
| LoRA | **0.5646** |
| Không fine-tune | 0.5405 |
| Full fine-tuning | 0.5357 |

Do đó nên tiếp tục giữ:

```yaml
r: 16
alpha: 32
dropout: 0.1
```

Tuy nhiên, nên dùng tốc độ học riêng:

- LoRA backbone: \(2\times10^{-5}\) hoặc \(5\times10^{-5}\);
- fusion và classifier: \(1\times10^{-4}\);
- spatial projection: \(5\times10^{-5}\).

Hiện tại tất cả tham số Phase 2 dùng cùng \(10^{-4}\), có thể làm LoRA thay đổi backbone nhanh hơn cần thiết.

---

# 5. Điều chỉnh nhánh high-resolution

Không nên loại bỏ high-resolution ngay vì nó cải thiện so với direct resize:

\[
\Delta\text{Macro-F1}=+0.0099,
\qquad
\Delta\text{Balanced Acc}=+0.0266.
\]

Tuy nhiên, letterbox đạt macro-F1 0.5781, cao hơn sparse-focal 0.5646. Vì vậy cần thử high-resolution dưới Phase-2-only thay vì kết luận từ mô hình hai pha.

## 5.1. Thay mean pooling bằng attention pooling

Hiện Phase 1 dùng:

\[
\overline{\mathbf v}_l
=
\frac{1}{8}\sum_{j=1}^{8}\mathbf v_{l,j}.
\]

Nên thay bằng:

\[
a_j=
\frac{
\exp(\mathbf g_v^\top W\mathbf v_{l,j}/\tau_l)
}{
\sum_k
\exp(\mathbf g_v^\top W\mathbf v_{l,k}/\tau_l)
},
\]

\[
\mathbf v_l^{\mathrm{attn}}
=
\sum_{j=1}^{8}a_j\mathbf v_{l,j}.
\]

Cách này ưu tiên tile tương thích với global context thay vì gán trọng số bằng nhau.

## 5.2. Tăng độ phân giải token cục bộ

Hiện mỗi tile được nén:

\[
196\text{ patch}\rightarrow1\text{ pooled token}.
\]

Có thể thử:

```yaml
local_pool_grid: 2
include_local_cls_token: true
num_visual_tokens: 20
```

Mỗi tile khi đó tạo:

\[
1+2^2=5\text{ token},
\]

và bốn tile tạo:

\[
4\times5=20\text{ local token}.
\]

Luồng fusion trở thành:

\[
[B,21,512].
\]

Đây là thay đổi đáng thử nếu tổn thương nhỏ thường bị mất khi pooling \(14\times14\rightarrow1\times1\).

## 5.3. Không tăng số tile trước

Tăng từ 4 lên 8 tile sẽ làm chi phí ViT tăng gần gấp đôi nhưng không bảo đảm tile mới có liên quan. Nên tăng thông tin giữ lại trong mỗi tile trước khi tăng số tile.

---

# 6. Đơn giản hóa hoặc kiểm soát cross-attention

Cross-attention không cải thiện macro-F1 so với concat, nhưng tăng Balanced Accuracy. Vì vậy không nên loại bỏ ngay.

Thí nghiệm quan trọng là:

\[
\text{Phase-2-only}
\times
\{\text{concat},\text{bidirectional}\}.
\]

Kết quả concat hiện tại vẫn dùng checkpoint Phase 1, nên chưa phải phép so sánh sạch với Phase-2-only.

Nếu giữ cross-attention, nên thêm residual gate:

\[
\mathbf h_v
=
\operatorname{LN}
\left(
\mathbf g_v+
\sigma(\gamma_v)\mathbf c_t
\right),
\]

\[
\mathbf h_t
=
\operatorname{LN}
\left(
\mathbf g_t+
\sigma(\gamma_t)\mathbf c_v
\right).
\]

Khởi tạo:

\[
\sigma(\gamma_v)=\sigma(\gamma_t)=0.1\text{ hoặc }0.25.
\]

Gate giúp mô hình chỉ sử dụng context từ phương thức còn lại khi context đó có lợi, thay vì luôn cộng toàn bộ attention output.

---

# 7. Nếu vẫn muốn giữ Phase 1

Nếu kiến trúc hai pha là một đóng góp bắt buộc, Phase 1 cần được thiết kế lại.

## 7.1. Giảm target similarity

Thay:

```yaml
target_similarity: 0.95
```

bằng các mức:

```yaml
target_similarity: 0.25
```

hoặc:

```yaml
target_similarity: 0.50
```

Không nên coi hai mẫu cùng lớp gần tương đương với cặp ảnh–bệnh sử thật.

## 7.2. Dùng hệ số local học được

Thay:

\[
\mathbf v=
\operatorname{Norm}
(\mathbf g_v+0.25\mathbf v_l)
\]

bằng:

\[
\mathbf v=
\operatorname{Norm}
\left(
\mathbf g_v+\sigma(a)\mathbf v_l
\right).
\]

Khởi tạo \(\sigma(a)=0.25\), sau đó cho mô hình tự điều chỉnh.

## 7.3. Bảo đảm positive pairs trong batch

Với batch 16 và 22 lớp, nhiều lớp không có mẫu cùng nhãn trong batch. Nên dùng class-aware batch sampler, chẳng hạn:

- tám lớp mỗi batch;
- hai mẫu cho mỗi lớp.

Khi đó batch vẫn có 16 mẫu nhưng mỗi mẫu có ít nhất một positive pair.

## 7.4. Thay hai pha tách biệt bằng loss phối hợp

Một phương án ổn định hơn:

\[
\mathcal L
=
\mathcal L_{\mathrm{CE}}
+
\lambda_{\mathrm{con}}
\mathcal L_{\mathrm{contrastive}},
\]

với:

\[
\lambda_{\mathrm{con}}\in\{0.05,0.1\}.
\]

Loss phân loại giữ nhiệm vụ chính, còn contrastive loss chỉ đóng vai trò regularization. Cách này giảm nguy cơ Phase 1 đưa backbone đến một vùng nghiệm không phù hợp với phân loại.

---

# 8. Thiết lập thực nghiệm tối thiểu nên chạy

Không cần thử một grid quá lớn. Sáu cấu hình sau đủ để xác định hướng tốt nhất:

| ID | Phase 1 | Preprocess | Fusion | Classifier |
|---|---|---|---|---|
| A | Không | Sparse-focal | Bidirectional | Centroid |
| B | Không | Sparse-focal | Bidirectional | Linear |
| C | Không | Sparse-focal | Concat | Linear |
| D | Không | Letterbox | Bidirectional | Linear |
| E | Có, target 0.50 | Sparse-focal | Bidirectional | Linear |
| F | Joint loss \(\lambda=0.1\) | Sparse-focal | Bidirectional | Linear |

Mỗi cấu hình chạy ba seed trên CTCH và BTXRD. Thứ tự lựa chọn:

1. Macro-F1 trung bình;
2. Balanced Accuracy;
3. bootstrap CI;
4. Macro-AUROC;
5. chi phí huấn luyện và suy luận.

# Khuyến nghị cuối cùng

Cấu hình có xác suất cải thiện cao nhất hiện tại là:

```text
BiomedCLIP
+ sparse-focal bốn tile
+ LoRA
+ bỏ Phase 1
+ bidirectional cross-attention có residual gate
+ linear hoặc learnable cosine classifier
+ effective-number weights
+ label smoothing 0.05
```

Tuy nhiên, thay đổi đầu tiên nên thực hiện chỉ là:

```text
Phase-2-only + linear head
```

và so sánh với:

```text
Phase-2-only + empirical centroid
```

Nếu linear head không cải thiện, giữ centroid và tập trung tối ưu high-resolution/fusion. Nếu linear head cải thiện macro-F1 trên cả CTCH và BTXRD, có thể xác định đầu empirical-centroid là nút thắt còn lại sau khi đã loại bỏ Phase 1.