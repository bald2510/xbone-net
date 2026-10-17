# Đánh giá thực nghiệm và đề xuất tên bài

## Mức độ thuyết phục hiện tại

Bài đã có nền tảng thực nghiệm hợp lý cho một nghiên cứu ứng dụng: hai bộ dữ liệu, sáu baseline, ba training seed, đối sánh Image-only, ablation về alignment/fusion, và số đo bộ nhớ/thời gian. Đây là nhận định biên tập dựa trên bản hiện tại; không phải dự báo chấp nhận. Phạm vi bài phù hợp AI for healthcare và explainable AI in applications trong [CFP RIVF 2026](https://rivf2026.org/call-for-papers.html).

Thông điệp có bằng chứng rõ nhất là: XBone-Net đạt Macro-F1 cạnh tranh với ít tham số cập nhật và ít peak allocated training memory hơn, đổi lại thời gian huấn luyện và latency tăng. Không nên chuyển thành tuyên bố nhanh hơn, ưu việt ở mọi metric hoặc đã sẵn sàng triển khai lâm sàng.

| Khía cạnh | Bằng chứng đã có | Khoảng trống còn lại |
|---|---|---|
| Phân loại | Mean ± SD của ba seed; CTCH và BTXRD; sáu baseline | Fixed split; chưa external patient-level validation; không dẫn đầu mọi metric |
| Chi phí | Trainable parameters, training memory/time, GFLOPs, latency, throughput | Profile cục bộ; chưa đo workflow thực tế |
| Thành phần | Phase-2-only và Concatenation; Image-only ở Bảng IV | Chưa tách sampler/loss; Image-only thay đổi nhiều thành phần cùng lúc |
| Ảnh và bệnh sử | IG-guided deletion so với random trên đủ 669 CTCH test cases, seed 42 | Chưa chứng minh tương tác liên phương thức, đúng ngữ nghĩa region–word hoặc lợi ích lâm sàng của giải thích |
| Tái lập | 51 prediction artifacts đã đối chiếu, mã nguồn và aggregate provenance | Dòng BTXRD Image-only mới được cung cấp dưới dạng aggregate; chưa tính lại từ ba prediction files |

## Có nên đưa thêm IG vào không?

Nên giữ IG định lượng trong bài; phiên bản hiện tại đã có Figure 2 gồm hai deletion curves từ đủ 669 ca. Kết quả này bổ trợ cho đóng góp phân tích cách mô hình sử dụng đầu vào, không thay thế ablation hoặc kiểm tra khái quát hóa.

Nếu thêm minh họa định tính, nên dùng một panel gọn với 2–3 ca có quy tắc chọn minh bạch: một ca đúng tiêu biểu, một ca multimodal đúng nhưng Image-only sai, và một ca multimodal sai. Chỉ dùng loại ca thứ hai nếu thực sự có trong các dự đoán được căn chỉnh. Mỗi ca cần ảnh và IG overlay, đoạn text với attribution, nhãn thật, dự đoán và xác suất; giữ cùng target/reference protocol. Không gọi vùng sáng là vùng tổn thương đã được xác thực khi chưa có annotation hoặc bác sĩ đối chiếu. Nêu rõ ví dụ minh họa không thay thế đánh giá toàn bộ test set.

Với giới hạn sáu trang, ưu tiên phần IG định lượng đã có. Chỉ thêm panel khi có thể rút phần diễn giải lặp lại và vẫn giữ hình đọc được; không ép thêm bằng cách thu nhỏ toàn bài. Lần cập nhật này giữ nguyên Figure 2, chưa tạo thêm heatmap hay số liệu mới.

## Thí nghiệm bổ sung có giá trị cao nhất

1. **Kiểm soát bệnh sử trên mô hình cố định:** bệnh sử đúng, thiếu và hoán đổi có kiểm soát. So sánh Macro-F1, per-class recall và thay đổi xác suất trên cùng test IDs. Thí nghiệm này trả lời trực tiếp mô hình tận dụng bối cảnh riêng của bệnh nhân hay dựa vào dấu hiệu nhãn trong text. Thiết kế trên validation trước khi chạy test; không chọn quy tắc hoán đổi theo kết quả test.
2. **Text-only / nội dung bệnh sử:** bổ sung text-only baseline và rà soát các từ ngữ tiết lộ chẩn đoán. Việc này đặc biệt quan trọng vì BTXRD sử dụng label-correlated synthetic text; lợi ích multimodal trên bộ này không tự chứng minh giá trị bệnh sử lâm sàng.
3. **Ablation được kiểm soát:** tách soft matching, class-aware sampler và class-balanced loss; giữ backbone, LoRA budget, fusion và training budget tương ứng để quy kết tác dụng từng thành phần.

External validation là hướng quan trọng về sau. Nếu thời gian hạn chế, kiểm soát bệnh sử mang thêm bằng chứng về luận điểm ảnh–text nhiều hơn chỉ bổ sung một hình heatmap.

## Tên bài đề xuất

1. **XBone-Net: Parameter-Efficient Radiograph Classification with Clinical History** — đề xuất ưu tiên cho track ứng dụng: ngắn, nêu tên mô hình, bài toán và đặc điểm thích nghi.
2. **XBone-Net: Class-Aware Alignment and Bidirectional Fusion for Radiograph–History Classification** — phù hợp nếu muốn nhấn mạnh hai thành phần phương pháp.
3. **Parameter-Efficient Adaptation of Biomedical Vision–Language Models for Radiograph–History Classification** — đặt trọng tâm ở hướng nghiên cứu, không đặt tên mô hình lên đầu.
4. **Adapting BiomedCLIP for Radiograph–History Classification: Performance, Cost, and Input Dependence** — nhấn mạnh nghiên cứu thực nghiệm và đánh đổi ứng dụng; thận trọng hơn về tính mới kiến trúc.

Không dùng “clinically validated”, “robust”, “interpretable diagnosis” hoặc “state-of-the-art” trong tiêu đề với bằng chứng hiện tại. Tác giả đã chọn phương án 1: **XBone-Net: Parameter-Efficient Radiograph Classification with Clinical History**. Tiêu đề này đã được áp dụng vào PDF và nguồn LaTeX. Abstract và keywords giữ nội dung đã cập nhật.
