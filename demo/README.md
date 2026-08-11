# XBone-Net Streamlit demo

Demo nhận một ảnh X-quang và một đoạn bệnh sử, sau đó:

1. tạo và cho phép xem global view, vùng tiền cảnh cùng bốn sparse-focal local
   tile theo cấu hình checkpoint;
2. suy luận phân loại CTCH 22 lớp từ fused embedding, hiển thị toàn bộ xác suất
   lớp và confidence (max softmax);
3. tính OOD score với detector khớp trên CTCH train và ngưỡng phân vị 95% được
   khóa từ CTCH validation-ID;
4. truy xuất các ảnh CTCH-train gần nhất bằng cosine similarity trong không gian
   global visual embedding;
5. tạo Integrated Gradients cho cả global view và local visual tokens, sau đó
   ánh xạ đồng thời hai kết quả về ảnh nguồn;
6. tạo Integrated Gradients có dấu cho từng token bệnh sử qua text encoder,
   cross-attention, fusion MLP và classifier.

Giao diện dùng theme sáng, logo Trường Đại học Khoa học Tự nhiên, bảng xác suất
nền trắng và hiển thị đồng thời global IG, local IG cùng text IG sau một lần
chạy mô hình.

Tạo hình minh họa ảnh–văn bản bằng checkpoint `proposed_v3`, seed 42:

```powershell
python tools\visualize\render_multimodal_ig.py --image-id 3021_img-84263-00001.jpg --experiment ctch/proposed/ours_xbone_net_v3 --seed 42 --device cuda --ig-steps 24 --output results\visualization\3021_img-84263-00001_proposed_v3_multimodal_ig.png
```

Lệnh tạo một ảnh tổng hợp, một tệp CSV chứa attribution có dấu của từng từ và
một tệp CSV chứa đường cong deletion/insertion. Deletion thay dần token quan
trọng bằng baseline, còn insertion khôi phục dần token từ baseline; AUC được
tính trên xác suất của lớp dự đoán. Màu cam biểu thị đóng góp ủng hộ, còn màu
xanh biểu thị đóng góp phản đối logit của lớp dự đoán. Cổng OOD không được áp
dụng mặc định khi dựng một mẫu CTCH đã biết; thêm `--enable-ood` khi feature
archive và ngưỡng OOD của đúng checkpoint đã sẵn sàng.

Chạy từ thư mục gốc của repository:

```powershell
python -m pip install -r demo\requirements.txt
python -m streamlit run demo\streamlit_app.py
```

Mặc định demo dùng checkpoint canonical tại
`checkpoints/ctch/proposed/ours_xbone_net/seed_<seed>/best_phase2.pth` và các
feature archive tương ứng trong `results/ctch/proposed/ours_xbone_net`.

Không dùng ảnh hoặc bệnh sử của người bệnh thật trên máy chủ công khai. Demo chỉ
phục vụ nghiên cứu, không phải thiết bị y tế.
