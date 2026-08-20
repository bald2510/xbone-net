# Phân loại tài liệu tham khảo dạng PDF

Các tệp PDF được phân loại dựa trên cây nội dung chính thức của luận văn (`main.tex`). Mỗi bài báo được lưu trữ duy nhất tại một vị trí tương ứng với vai trò của nó trong khóa luận.

## 1. Thư mục `Selected`

Thư mục này chứa đúng **8 tài liệu** được trích dẫn trực tiếp trong **Chương 3 (Phương pháp đề xuất)**, đóng vai trò là các thành phần cốt lõi của hệ thống XBone-Net:
- `[54]` BiomedCLIP: Mô hình nền tảng ngôn ngữ-thị giác y sinh.
- `[20]` LoRA: Kỹ thuật thích nghi tham số hiệu quả hạng thấp.
- `[9]` Vision Transformer (ViT-B/16): Bộ mã hóa thị giác toàn cục và cục bộ.
- `[49]` MedCLIP: Hàm mất mát khớp ngữ nghĩa (Semantic Matching Loss) cho Pha 1.
- `[5]` Class-Balanced Loss: Xử lý mất cân bằng lớp hiệu dụng cho Pha 2.
- `[43]` Prototypical Networks: Bộ phân lớp dựa trên tâm lớp thực nghiệm.
- `[44]` Deep Nearest Neighbors (kNN): Phát hiện dữ liệu ngoài phân phối (Multi-modal Ensemble OOD).
- `[45]` Integrated Gradients (IG): Phương pháp giải thích và gán quyền đặc trưng đa phương thức.

## 2. Thư mục `Methods`

Chứa 52 tài liệu tham khảo được phân chia thành 6 nhóm chủ đề:
- **`Dataset`** (2 PDF): Bộ dữ liệu và công cụ gán nhãn (`russell2008labelme`, `Yao2025`).
- **`Med-MLLLM`** (4 PDF): Mô hình ngôn ngữ lớn đa phương thức y khoa và mô hình tổng quát (`li_2023_llavamed`, `moor_2023_medflamingo`, `OpenAI2024GPT4oSystemCard`, `MED-PALM-M`).
- **`Med-VLMs`** (13 PDF): Các mô hình thị giác-ngôn ngữ y khoa, mô hình nền tảng và thích nghi tham số (`TieNet`, `ViLBERT`, `GLoRIA`, `CLIP`, `ConVIRT`, `BioViL`, `PubMedCLIP`, `CLIPath`, `IRENE`, `LCBB`, `FTB`, `BiomedCoOp`, `PLIP`).
- **`Metrics`** (13 PDF): Độ đo phân lớp, hiệu chỉnh, kiểm định thống kê, độ đo phân cụm và độ đo giải thích (`Holm1979Sequential`, `Davies1979ClusterSeparation`, `Rousseeuw1987Silhouettes`, `EfronTibshirani1993Bootstrap`, `Good2005PermutationBootstrap`, `Fawcett2006ROC`, `Davis2006PR`, `Brodersen2010BalancedAcc`, `Canziani2017PracticalDNN`, `Guo2017Calibration`, `Petsiuk2018RISE`, `Reddi2020MLPerfInference`, `DeYoung2020ERASER`).
- **`OOD Detection`** (5 PDF): Phương pháp phát hiện OOD và hàm mất mát tâm lớp (`CenterLoss`, `Hendrycks2017MSP`, `Lee2018MahalanobisOOD`, `Liang2018ODIN`, `Liu2020EnergyOOD`).
- **`Scratch`** (15 PDF): Mạng nơ-ron nền tảng, học tự giám sát, tối ưu hóa, kỹ thuật tiền huấn luyện và công cụ (`ResNet`, `DenseNet`, `Vaswani2017Attention`, `Oord2018CPC`, `Houlsby2019Adapters`, `Jain2019AttentionNotExplanation`, `Loshchilov2019AdamW`, `Wiegreffe2019AttentionNotNotExplanation`, `He2020MoCo`, `XMei`, `Chen2020SimCLR`, `DeepTranslator`, `Gu2021PubMedBERT`, `Lester2021PromptTuning`, `ParaphraseMultilingualMiniLM`).
