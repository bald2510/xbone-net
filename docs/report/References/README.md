# Phân loại tài liệu tham khảo dạng PDF

Các tệp PDF được phân loại dựa trên cây nội dung chính thức của luận văn (`main.tex`). Mỗi bài báo được lưu trữ duy nhất tại một vị trí tương ứng với vai trò của nó trong khóa luận.

## 1. Thư mục `Selected`

Thư mục này chứa đúng **8 tài liệu** được trích dẫn trực tiếp trong **Chương 3 (Phương pháp đề xuất)**, đóng vai trò là các thành phần cốt lõi của hệ thống XBone-Net:
- `[41]` BiomedCLIP: Mô hình nền tảng ngôn ngữ-thị giác y sinh.
- `[14]` LoRA: Kỹ thuật thích nghi tham số hiệu quả hạng thấp.
- `[8]` Vision Transformer (ViT-B/16): Bộ mã hóa thị giác toàn cục và cục bộ.
- `[37]` MedCLIP: Hàm mất mát khớp ngữ nghĩa (Semantic Matching Loss) cho Pha 1.
- `[4]` Class-Balanced Loss: Xử lý mất cân bằng lớp hiệu dụng cho Pha 2.
- `[32]` Prototypical Networks: Bộ phân lớp dựa trên tâm lớp thực nghiệm.
- `[33]` Deep Nearest Neighbors (kNN): Phát hiện dữ liệu ngoài phân phối (Multi-modal Ensemble OOD).
- `[34]` Integrated Gradients (IG): Phương pháp giải thích và gán quyền đặc trưng đa phương thức.

## 2. Thư mục `ByCitation`

Chứa các tài liệu tham khảo được trích dẫn trong các chương còn lại (Chương 1, 2, 4 và Phụ lục), phân chia thành các nhóm chủ đề:
- **`Dataset`** (2 PDF): Bộ dữ liệu và công cụ gán nhãn (`russell2008labelme`, `Yao2025`).
- **`Med-VLMs`** (10 PDF): Các mô hình thị giác-ngôn ngữ y khoa và thích nghi nền tảng (`ConVIRT`, `GLoRIA`, `PLIP`, `BioViL`, `CLIPath`, `FTB`, `LCBB`, `BiomedCoOp`, `CLIP`, `PubMedCLIP`).
- **`Med-MLLLM`** (3 PDF): Mô hình ngôn ngữ lớn đa phương thức y khoa (`moor_2023_medflamingo`, `li_2023_llavamed`, `MED-PALM-M`).
- **`OOD Detection`** (5 PDF): Phương pháp phát hiện OOD và hàm mất mát tâm lớp (`Hendrycks2017MSP`, `Liang2018ODIN`, `Liu2020EnergyOOD`, `Lee2018MahalanobisOOD`, `CenterLoss`).
- **`Explainable AI`** (2 PDF): Mô hình chú ý và gióng hàng giải thích (`TieNet`, `IRENE`).
- **`Scratch`** (3 PDF): Mạng tích chập nền tảng và mô hình xây dựng từ đầu (`ResNet`, `DenseNet`, `XMei`).
- **`Metrics`** (7 PDF): Độ đo, hiệu chỉnh, xếp hạng và benchmark (`Brodersen2010BalancedAcc`, `Canziani2017PracticalDNN`, `Davis2006PR`, `Fawcett2006ROC`, `Guo2017Calibration`, `Reddi2020MLPerfInference`, `Rousseeuw1987Silhouettes`).

## 3. Tệp danh mục

- `classification.csv`: Danh mục phân loại chi tiết 44 tài liệu theo số thứ tự trích dẫn, nhóm, đường dẫn tệp và lý do phân loại.
- `missing.csv`: Danh sách tài liệu dạng tài nguyên trực tuyến hoặc bài báo đóng (`Davies1979ClusterSeparation`, `DeepTranslator`, `OpenAI2024GPT4oSystemCard`, `ParaphraseMultilingualMiniLM`).
- `selection_summary.json`: Tóm tắt thống kê số lượng PDF theo từng nhóm.
