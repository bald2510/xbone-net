# XBone-Net Streamlit demo

Demo nhận một ảnh X-quang và một đoạn bệnh sử, sau đó:

1. tạo và cho phép xem global view, vùng tiền cảnh cùng bốn sparse-focal local
   tile theo cấu hình checkpoint;
2. khớp temperature scaling trên CTCH validation-ID của đúng checkpoint, sau
   đó dùng calibrated maximum softmax probability làm confidence cho mẫu ID;
3. tính OOD score với detector khớp trên CTCH train và ngưỡng phân vị 95% được
   khóa từ CTCH validation-ID;
4. nếu phát hiện OOD, chỉ hiển thị cảnh báo và các ảnh CTCH-train gần nhất,
   không gán nhãn bệnh, không hiển thị confidence hoặc bảng xác suất;
5. nếu mẫu được chấp nhận là ID, hiển thị lớp dự đoán, confidence hiệu chỉnh,
   bảng xếp hạng theo softmax gốc và các ảnh tham chiếu gần nhất;
6. tạo Integrated Gradients cho cả global view và local visual tokens, sau đó
   ánh xạ đồng thời hai kết quả về ảnh nguồn;
7. tạo Integrated Gradients có dấu cho từng token bệnh sử qua text encoder,
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

Demo chấm điểm được khóa ở seed 42 và đọc trực tiếp ba tệp đã đóng gói:

```text
demo/artifacts/best_phase2.pth
demo/artifacts/features/ctch_train.npz
demo/artifacts/features/ctch_val.npz
```

Checkpoint dùng để suy luận; archive train dùng để khớp bộ phát hiện OOD và
tìm mẫu gần nhất; archive validation dùng để hiệu chỉnh confidence cùng ngưỡng
OOD. Ba tệp phải được sao chép đầy đủ khi chuyển repository sang máy mới.
Checkpoint có kích thước khoảng 781 MiB nên không thể đẩy lên GitHub thông
thường nếu không sử dụng Git LFS hoặc một kênh chuyển tệp lớn khác.

Không dùng ảnh hoặc bệnh sử của người bệnh thật trên máy chủ công khai. Demo chỉ
phục vụ nghiên cứu, không phải thiết bị y tế.
