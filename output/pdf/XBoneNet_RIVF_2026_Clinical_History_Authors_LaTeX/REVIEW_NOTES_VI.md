# Bản chỉnh sửa XBone-Net cho RIVF 2026

Ngày đối chiếu: 15/09/2026. Bản gốc là hai tệp người dùng cung cấp trong Downloads; không thay đổi hai tệp đó. Bản sửa sử dụng kết quả đã có, không huấn luyện thêm hoặc tạo dự đoán mới.

## Định vị bài báo

Bài được định vị là nghiên cứu ứng dụng: thích nghi mô hình nền tảng cho phân loại X-quang kết hợp bệnh sử khi dữ liệu mất cân bằng và bộ nhớ huấn luyện hạn chế. Điểm nhấn là bằng chứng về hiệu năng, chi phí đánh đổi và cách mô hình sử dụng hai đầu vào. Đây là cách trình bày phù hợp phạm vi AI Applications / AI for healthcare, không phải bảo đảm chấp nhận.

[CFP chính thức RIVF 2026](https://rivf2026.org/call-for-papers.html) quy định tiếng Anh, tối đa sáu trang, IEEE conference, A4 và PDF 1.6. CFP hiện ghi hạn cuối 15/09/2026, không gia hạn thêm; cần kiểm tra giờ đóng cổng trực tiếp trên EDAS.

## Các thay đổi chính

| Phần | Điều chỉnh |
|---|---|
| Abstract / Introduction | Làm rõ bài toán ứng dụng, khoảng trống thích nghi với bệnh sử, ba đóng góp và kết quả định lượng. Phân biệt hiệu quả cập nhật tham số với tốc độ chạy. |
| Related Work | Viết theo tiến trình từ kết hợp dữ liệu lâm sàng đến pretraining và thích nghi hiệu quả. Bổ sung BiomedCoOp (CVPR 2025), chỉ rõ đây là công trình liên quan chưa được chạy như baseline trong nghiên cứu. |
| Methods | Chuẩn hóa ký hiệu, phân biệt effective number và inverse weight; mô tả chuẩn hóa token, bidirectional attention, class-balanced smoothed loss, hai giai đoạn offline và suy luận online. |
| Experimental Setup / Datasets | Bổ sung cấu hình, nguồn bệnh sử, phép chia dữ liệu, phân bố số mẫu lớp và giới hạn văn bản tổng hợp BTXRD. |
| Classification | Bảng hai bộ dữ liệu cùng thang phần trăm, mean ± sample SD; sáu baseline nằm riêng với hai cấu hình XBone-Net. Tô đậm đúng cực đại từng cột, không tô đậm phương pháp đề xuất ở mọi metric. |
| Cost / Ablation | Bổ sung GPU-hours và peak allocated training memory cùng latency, throughput, GFLOPs; giữ ablation theo kết quả hiện có, không thêm kiểm định thống kê. |
| Image–text analysis | IG trên toàn bộ 669 CTCH test cases, seed 42, gồm cả dự đoán sai; giữ target class khi can thiệp và giải thích đúng ý nghĩa conditional sensitivity. |
| Figures / Conclusion | Vẽ sơ đồ vector và đường IG bằng LaTeX; kết luận gắn với bằng chứng, nêu hướng đánh giá ngoài cơ sở, kiểm soát bệnh sử và giảm chi phí. |

## Kết quả đã đối chiếu

Đã tính lại Accuracy, Balanced Accuracy và Macro-F1 từ 51 tệp dự đoán theo seed. Các phương pháp được căn chỉnh theo image ID và nhãn; kiểm tra ID duy nhất và dự đoán bằng argmax của xác suất. Trong bảng bài báo, các chỉ số được đổi sang phần trăm nhưng Abstract giữ thang 0–1.

| Đại lượng | Giá trị sử dụng |
|---|---|
| CTCH XBone-Net Macro-F1 | 0.5727 ± 0.0193, ba seed |
| BTXRD XBone-Net Macro-F1 | 0.6240 ± 0.0118, ba seed |
| CTCH trainable parameters | 8.45M / 204.35M, khoảng 4.13% |
| CTCH peak allocated memory | 1455.0 ± 43.0 MiB so với 3816.8 ± 0.0 MiB của full BiomedCLIP tuning |
| CTCH training cost | 0.8542 ± 0.0925 so với 0.3411 ± 0.0800 GPU-hours |
| IG image deletion AUC | targeted 0.2997; random 0.3405 |
| IG text deletion AUC | targeted 0.2811; random 0.3588 |
| Whole-source replacement probability drop | ảnh 0.3187; text 0.2725 |

Các số liệu giảm 61.9% bộ nhớ và tăng 2.50 lần thời gian được tính từ trung bình ba seed. Bộ nhớ là peak allocated của các giai đoạn huấn luyện, không phải dung lượng VRAM tối thiểu bảo đảm chạy được hệ thống. Thời gian inference là forward profiling, không bao gồm luồng tiếp nhận ca bệnh.

`evidence/audit.json` lưu các kết quả tính lại, phân bố lớp, nguồn và SHA-256. Nguồn chính là các tệp `analysis/features/*_test.npz`, `training_summary.json`, `analysis/explainability_full_test/{summary,samples}.json`, cấu hình/mã nguồn và báo cáo khóa luận đang hoạt động. Các tệp đóng gói không chứa bệnh sử hoặc ảnh bệnh nhân gốc.

## Những điểm cần tác giả xử lý trước khi nộp

1. **BTXRD Image-only đã được bổ sung theo số liệu tác giả cung cấp ở yêu cầu tiếp theo:** Accuracy 0.6423 ± 0.0154, Balanced Accuracy 0.3267 ± 0.0169 và Macro-F1 0.4021 ± 0.0257. Bảng IV dùng phần trăm nên hiển thị 64.23 ± 1.54, 32.67 ± 1.69 và 40.21 ± 2.57. Đây là dữ liệu do tác giả cung cấp, chưa được tính lại độc lập từ dự đoán đủ ba seed trong kho hiện tại. Nguồn được ghi riêng trong `evidence/author_supplied_results.json`, không nhập lẫn vào 51 tệp dự đoán đã kiểm chứng. Không còn ô thiếu kết quả trong Bảng IV.
2. **Thông tin quản trị dữ liệu CTCH:** cần xác nhận phê duyệt hoặc miễn trừ đạo đức nghiên cứu, sự đồng ý/miễn đồng ý áp dụng, và quyền công bố kết quả. Thỏa thuận bảo mật và de-identification không tự thay thế các xác nhận đó. Bản sửa không tự điền số phê duyệt hoặc tuyên bố đã được duyệt.
3. **Bệnh sử và thời điểm dự đoán:** cần xác nhận các trường thực sự có sẵn trước chẩn đoán đích, cách dịch và rà soát chất lượng. Việc loại trường chẩn đoán trực tiếp chưa loại trừ thông tin chẩn đoán còn trong văn bản bệnh sử. BTXRD synthetic text liên quan nhãn nên chỉ được diễn giải trong thiết lập đó.
4. **Đối sánh và ablation:** các chênh lệch là kết quả quan sát trên fixed split với ba training seed. Nghiên cứu chưa chứng minh cải thiện ở mọi lớp hiếm hoặc qua cơ sở độc lập. Image-only đồng thời thay đổi text access, alignment và fusion nên không đo riêng tác dụng của bệnh sử.
5. **IG:** seed 42 có 669 ca và 403 dự đoán đúng; phân tích không lọc chỉ lấy ca đúng. Không suy ra lesion localization, quan hệ nhân quả hoặc đúng ngữ nghĩa region–word từ deletion curves. Kiểm tra hội tụ theo số bước tích phân, reference và seed là hướng cần làm nếu muốn khẳng định explanation stability. Phân tích hiện tại không thay thế đánh giá của bác sĩ.

Phần tác giả đã được trình bày theo mẫu người dùng cung cấp: tên theo thứ tự tên–họ, chỉ số 1,2 cho hai dòng đơn vị công tác hiện có và bốn email nvlbthanh22@clc.fitus.edu.vn, ngkiet22@clc.fitus.edu.vn, lqngoc@fit.hcmus.edu.vn, dttha@fit.hcmus.edu.vn. Các email được nhóm theo tên miền. Đã thêm khai báo vai trò OpenAI Codex đối với phần văn bản, LaTeX và hình dựng từ kết quả có sẵn, theo [hướng dẫn IEEE về nội dung do AI hỗ trợ tạo](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/).

## Kiểm tra nguồn công trình liên quan

- [ConVIRT, PMLR](https://proceedings.mlr.press/v182/zhang22a.html): MURA AUC 89.0% ở full-data fine-tuning, Table 1(b).
- [MedCLIP, EMNLP 2022](https://aclanthology.org/2022.emnlp-main.256/): Table 2 báo cáo GLoRIA 0.7981 và MedCLIP 0.8075 trên RSNA; phần 4.6 mô tả frozen image encoder và học linear head.
- [BiomedCoOp, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Koleilat_BiomedCoOp_Learning_to_Prompt_for_Biomedical_Vision-Language_Models_CVPR_2025_paper.html), [bản toàn văn tác giả](https://arxiv.org/html/2411.15232v2): Table 1, 11 datasets, 16-shot accuracy 72.42 ± 1.69%. Không xem đây là kết quả chạy trên CTCH/BTXRD.

## Phạm vi hoàn tất

Bản sửa đã được biên dịch thành sáu trang A4, PDF 1.6, giữ cỡ chữ IEEE mặc định. Đã kiểm tra bảng, công thức, sơ đồ, đường IG và tài liệu tham khảo trên các trang render. Không có citation/reference chưa xác định hoặc overfull box trong lần biên dịch cuối. Các cảnh báo underfull về dàn chữ không làm tràn nội dung. Chưa nộp EDAS, chưa chạy IEEE PDF eXpress và chưa thực hiện các thí nghiệm mới nêu trên.
