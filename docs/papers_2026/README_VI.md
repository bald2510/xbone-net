# Hai bản thảo XBone-Net cho SOICT 2026 và RIVF 2026

Ngày đối chiếu yêu cầu: **06/09/2026**. Hai bản viết bằng tiếng Anh và là **hai phương án trình bày cùng công trình**, chưa được gửi đến hội nghị nào.

## Tệp để đọc và chỉnh sửa

- `soict/main.tex`: bản đầy đủ, nhấn mạnh phương pháp, cơ sở của hàm mất mát, bảng đối chiếu nghiên cứu trước và suy luận thống kê.
- `rivf/main.tex`: bản cô đọng cho IEEE, nhấn mạnh phân loại đa phương thức và đánh đổi giữa số tham số huấn luyện với chi phí suy luận.
- PDF cuối ở `output/pdf/` tại gốc repository: `XBoneNet_SOICT_2026_Draft.pdf` và `XBoneNet_RIVF_2026_Draft.pdf`.
- `XBoneNet_2026_LaTeX.zip`: nguồn hai bài, bibliography và hướng dẫn; không chứa ảnh bệnh nhân, văn bản bệnh án hay checkpoint.
- `evidence/classification_aggregate.csv`: số liệu cuối đưa vào bài.
- `evidence/classification_per_seed.csv`: 51 lượt đánh giá đã tính lại.
- `evidence/source_conflicts.csv`: 44 ô số liệu khác nhau giữa dự đoán lưu và `metrics.json` cũ; không sửa các artifact gốc.
- `evidence/statistics_verification.json`: kiểm tra 42 dòng thống kê ghép cặp có trung bình khớp dự đoán.
- `evidence/pdf_validation.json`: số trang, kích thước trang, font và lỗi biên dịch.

Các bản này là bản thảo để tác giả đọc, sửa và hoàn tất thông tin. Chưa thể gọi là hồ sơ sẵn sàng nộp vì email tác giả liên hệ và một số thông tin hành chính chưa được cung cấp.

## Yêu cầu hội nghị đã đối chiếu

| Tiêu chí | SOICT 2026 | RIVF 2026 |
|---|---|---|
| Mẫu | Springer CCIS/LNCS, lớp `llncs` chính thức tải từ Springer | `IEEEtran`, tùy chọn `conference,a4paper`; PDF 1.6 |
| Ngôn ngữ / tệp nộp | Tiếng Anh / PDF | Tiếng Anh / PDF |
| Giới hạn | 12 trang, không tính tài liệu tham khảo | Tối đa 6 trang; bản này tính cả tài liệu tham khảo trong giới hạn |
| Bản hiện tại | 12 trang nội dung + 1 trang tài liệu tham khảo | 5 trang tổng cộng |
| Tác giả | Single-blind, phải ghi tên và cơ quan | CFP không nêu yêu cầu ẩn danh; bản này có tên và cơ quan |
| Số trang in trên PDF | Không in số trang theo yêu cầu SOICT | Không in số trang theo mẫu conference |
| Kênh nộp | EasyChair, theo liên kết trên website | EDAS N35414 |
| Hạn abstract | 09/09/2026 | Không thấy hạn abstract riêng trên CFP chính |
| Hạn bài đầy đủ | 16/09/2026 | 15/09/2026, ghi là hạn cuối không gia hạn |
| Ngày hội nghị | 04–05/12/2026 | 18–20/12/2026 |

Nguồn chính thức: [SOICT paper submission](https://soict.org/submission/paper-submission/), [Springer author instructions and template](https://link.springer.com/series/558/information-for-authors-and-editors), [RIVF Call for Papers](https://rivf2026.org/call-for-papers.html). Hạn giờ và múi giờ đóng cổng cần kiểm tra trực tiếp trên hệ thống nộp bài; website được đọc không xác định rõ thông tin này. CFP RIVF ghi camera-ready 11/11/2026; không lấy mốc ở trang Doctoral Consortium cho regular paper.

Trang tổng quan CFP của SOICT còn tiêu đề cũ “SOICT 2024”, vì vậy định dạng và lịch được lấy từ trang **Paper submission 2026** cùng poster 2026. Không dùng mẫu ACM của các năm trước. Bộ mẫu tải trực tiếp từ Springer ngày 06/09/2026 chứa `llncs.cls` phiên bản 2.24 và `splncs04.bst`; bản SOICT dùng bibliography theo chính bộ mẫu này. Mẫu giữ nguyên vùng chữ và cỡ chữ mặc định, dùng font Latin Modern dạng outline để tránh font Type 3.

## Cấu trúc theo yêu cầu của bạn

1. **Abstract:** bối cảnh → tiến triển/khoảng trống → cách tiếp cận → kết quả và giới hạn.
2. **Introduction:** giới thiệu ngắn tiến trình nghiên cứu, ba đóng góp C1–C3 và phạm vi đánh giá.
3. **Related Works:** mô tả sự phát triển từ học đối sánh toàn cục, tương tác cục bộ, khớp ngữ nghĩa đến tiền huấn luyện quy mô lớn. Bảng có đủ nguyên lý, phương pháp, Data/Measure/Results, Pros và Cons.
4. **Methods:** framework offline–online, LoRA, soft semantic targets, cross-attention, linear head, weighted smoothed CE và thuật toán. Cơ sở khoa học của các thành phần được giải thích trước khi bàn tính hữu ích thực nghiệm.
5. **Experimental Results:** bảng mục đích thử nghiệm gắn với đóng góp; so sánh sáu baseline, các biến thể nội bộ, thống kê và chi phí. Có bàn luận trade-off và hạn chế.

Các thành phần là sự tích hợp các phương pháp đã có cho bài toán này. Bài không gọi LoRA, semantic matching hay cross-attention là phát minh mới của nhóm.

## Những lựa chọn khoa học cần giữ khi sửa

- Kiến trúc tương ứng với dự đoán đã chọn là **BiomedCLIP native image tokens + LoRA rank 16 + bidirectional cross-attention + linear head**. Không lấy mô tả bốn ảnh crop, sparse-focal branch hoặc centroid classifier từ `docs/paper/main.tex` cũ.
- Mọi số liệu phân loại được tính lại từ các `analysis/features/{dataset}_test.npz`, kiểm tra trùng tập image ID và nhãn. Gồm 30 lượt CTCH và 21 lượt BTXRD; không chạy lại huấn luyện.
- CTCH: 2.265/315/669 mẫu train/validation/test, 22 lớp, không có patient key chung giữa các phần chia. BTXRD: 2.621/375/750 ảnh, 10 lớp, chia ở mức ảnh.
- XBone-Net đạt CTCH Macro-F1 0,5727 ± 0,0193; BTXRD 0,6240 ± 0,0118. Dấu ± là độ lệch chuẩn mẫu của ba seed, không phải CI.
- CI và p-value lấy từ bộ phân tích ghép cặp đã có; đã xác nhận trung bình khớp các dự đoán đang dùng. Không chạy lại 10.000 lượt resampling trong tác vụ viết bài này.
- So sánh Macro-F1 với BiomedCLIP, CLIP, Phase 2 only và concatenation chưa đủ bằng chứng kết luận vượt trội. Không đổi cách viết này thành “state of the art” hay “significantly outperforms”.
- Image-only control hiện tại bỏ cả văn bản, Phase 1 và fusion. Mức chênh 0,1914 Macro-F1 không phải ước lượng cô lập tác dụng bệnh sử.
- Số đo GFLOPs/tốc độ/bộ nhớ chỉ thuộc **seed 42**, BF16, RTX 5060 Ti, batch 1, 20 đầu vào, mỗi đầu vào 10 warm-up và 10 forward đo thời gian. Đó không phải trung bình ba seed hoặc độ trễ workflow lâm sàng.
- BTXRD dùng văn bản sinh có liên hệ với metadata nhóm bệnh. Không diễn giải như bệnh sử thực tế hoặc external validation của CTCH.
- Các báo cáo OOD đang có nhiều phiên bản cohort và báo cáo tổng hợp bị ghi đè giữa mô hình. Hai bài chủ động giới hạn đóng góp ở phân loại, không đưa số OOD hay IG cũ vào để tăng số kết quả. Điều này không thay đổi dữ liệu OOD hoặc protocol gốc.
- Biểu thức logit scale Phase 1 dùng giá trị tiền huấn luyện cố định trong cấu hình LoRA hiện tại, không mô tả là tham số được tối ưu thêm.

## Nguồn của Related Works

- ConVIRT: [bài PMLR, Table 1(b)](https://proceedings.mlr.press/v182/zhang22a/zhang22a.pdf): MURA full-data fine-tuning AUC 89,0%.
- GLoRIA và MedCLIP: [MedCLIP, Table 2](https://aclanthology.org/2022.emnlp-main.256.pdf): RSNA linear-classification accuracy 79,81% và 80,75%. Số của GLoRIA là **đánh giá lại trong bài MedCLIP**, đã chú thích rõ, không giả là lấy từ thí nghiệm gốc GLoRIA.
- BiomedCLIP: [bản đầy đủ v3](https://arxiv.org/html/2303.00915v3), Figure 3 và mô tả linear probing; thông tin trích dẫn bản NEJM lấy theo [model card chính thức Microsoft](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224). Model card ghi năm 2024 cho DOI thuộc tập 2 số 1; nên kiểm tra metadata publisher lần cuối trước khi nộp.
- BIOMEDICA: [bài của tác giả](https://arxiv.org/abs/2501.07171): 40-task evaluation và mức cải thiện zero-shot trung bình 6,56% do bài đó báo cáo. Không so trực tiếp với CTCH/BTXRD.
- BTXRD: [bài dữ liệu Scientific Data](https://www.nature.com/articles/s41597-024-04311-y).

## Còn cần tác giả hoàn tất trước khi nộp

1. Xác nhận thứ tự tên, cơ quan, tên tiếng Anh và **email tác giả liên hệ**. Danh sách hiện lấy từ bài cũ: Nguyen Van Le Ba Thanh, Nguyen Gia Kiet, Ly Quoc Ngoc, Do Thi Thanh Ha. Không tự suy ra email hay ORCID. Springer yêu cầu email của corresponding author.
2. Điền thông tin chấp thuận nghiên cứu/miễn xét duyệt và quyền công bố dữ liệu CTCH phù hợp hồ sơ thật. Thỏa thuận bảo mật không tự thay thế chấp thuận đạo đức; bản thảo không bịa số phê duyệt.
3. Hoàn tất funding, acknowledgments, competing interests và xác nhận đóng góp của mọi tác giả. Bản SOICT đánh dấu khai báo lợi ích đang chờ xác nhận.
4. Giữ và chỉnh khai báo sử dụng AI theo đúng phần tác giả thực sự giữ lại. Hai bài đã có disclosure, vì AI hỗ trợ viết nhiều phần của bản thảo. Tham chiếu [IEEE submission policies](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/) và [Springer AI guidance](https://group.springernature.com/gp/group/ai/ai-guidance-for-our-researchers-and-communities).
5. Chọn **một hội nghị để gửi công trình này trong một thời điểm**. SOICT và RIVF đều yêu cầu không đang được xét duyệt nơi khác. Đổi tiêu đề, độ dài hay mẫu trình bày không làm hai bản trở thành hai công trình độc lập. Nếu muốn nộp cả hai, cần tách câu hỏi nghiên cứu và đóng góp thực chất, bổ sung kết quả và rà soát mức trùng lặp.
6. Đọc toàn văn và kiểm tra cổng nộp bài. Kiểm tra PDF nội bộ đã đạt không thay thế xác nhận của EasyChair/EDAS, IEEE PDF eXpress nếu được yêu cầu ở giai đoạn camera-ready, hay quyết định chấp nhận của hội nghị.

## Biên dịch và kiểm tra

Trong mỗi thư mục `soict` hoặc `rivf`, chạy `pdflatex main.tex`, `bibtex main`, rồi `pdflatex main.tex` hai lần. Có thể tải thư mục tương ứng lên Overleaf và chọn `main.tex`. Bản SOICT kèm lớp và bibliography chính thức; bản RIVF cần `IEEEtran` có sẵn trong các bản TeX phổ biến.

`build.ps1` dành cho cài đặt MiKTeX hiện tại trên máy này. `audit_evidence.py` và `prepare_assets.py` dùng môi trường Python nghiên cứu; không cần chạy để chỉnh câu chữ. Các bảng đã được chèn trực tiếp vào `main.tex` để từng bài là một tài liệu chỉnh sửa độc lập. Nếu cập nhật kết quả, phải đồng bộ lại những hàng nội tuyến này sau khi chạy audit; không chỉ thay CSV.

Các PDF đã được render và kiểm tra toàn bộ trang. Kết quả kiểm tra hiện tại: không lỗi tràn hộp, không trích dẫn hoặc tham chiếu chưa xác định, chỉ dùng font Type 1. Bản SOICT không có số trang/running heads; bản RIVF đúng A4 và PDF 1.6. Mọi thay đổi thêm sau này cần biên dịch lại và kiểm tra giới hạn trang.
