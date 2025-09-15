# Hướng dẫn tải files từ HuggingFace cho Image Prompt Demo

## 📋 Danh sách files cần tải

Để chạy config `yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py`, bạn cần tải các files sau:

### 1. YOLO-World Model Weights
```
URL: https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth
Kích thước: ~1.4GB
Đường dẫn server: pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth
```

### 2. OpenAI CLIP ViT-Base-Patch32 Model
Thư mục: `pretrained_models/open-ai-clip-vit-base-patch32/`

| File | URL | Kích thước |
|------|-----|------------|
| config.json | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json | ~1KB |
| pytorch_model.bin | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin | ~600MB |
| preprocessor_config.json | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json | ~1KB |
| tokenizer_config.json | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json | ~1KB |
| tokenizer.json | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json | ~2MB |
| vocab.json | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json | ~1MB |
| merges.txt | https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt | ~500KB |

**Tổng dung lượng: ~2GB**

## 🚀 Cách tải trên máy cá nhân

### Phương án 1: Sử dụng script Python tự động
```bash
# Copy file download_models_for_image_prompt.py về máy cá nhân
python download_models_for_image_prompt.py
```

### Phương án 2: Tải thủ công bằng wget/curl
```bash
# Tạo thư mục
mkdir -p yolo_world_models/pretrained_models/open-ai-clip-vit-base-patch32

# Tải YOLO-World model
wget -O yolo_world_models/pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth \
  "https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth"

# Tải CLIP model files
cd yolo_world_models/pretrained_models/open-ai-clip-vit-base-patch32
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt
```

### Phương án 3: Sử dụng git lfs (nếu có git)
```bash
# Clone CLIP model repository
git clone https://huggingface.co/openai/clip-vit-base-patch32
mv clip-vit-base-patch32 yolo_world_models/pretrained_models/open-ai-clip-vit-base-patch32

# Tải YOLO-World model riêng
wget -O yolo_world_models/pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth \
  "https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth"
```

## 📤 Upload lên server qua SSH

### 1. Nén files (tùy chọn)
```bash
# Nén để upload nhanh hơn
tar -czf yolo_world_models.tar.gz yolo_world_models/
```

### 2. Upload bằng SCP
```bash
# Upload thư mục
scp -r yolo_world_models/ username@server_ip:/home/quangnh58/Segment_logo/YOLO-World/

# Hoặc upload file nén
scp yolo_world_models.tar.gz username@server_ip:/home/quangnh58/Segment_logo/YOLO-World/
# Sau đó giải nén trên server:
# tar -xzf yolo_world_models.tar.gz
```

### 3. Upload bằng rsync (khuyến nghị)
```bash
# Sync với progress và resume capability
rsync -avz --progress yolo_world_models/ username@server_ip:/home/quangnh58/Segment_logo/YOLO-World/pretrained_models/
```

### 4. Sử dụng SFTP
```bash
sftp username@server_ip
put -r yolo_world_models/pretrained_models/* /home/quangnh58/Segment_logo/YOLO-World/pretrained_models/
```

## 🔧 Cấu trúc thư mục cuối cùng trên server

**⚠️ LƯU Ý QUAN TRỌNG:** Dựa trên config file, đường dẫn thực tế là:

```
/home/quangnh58/Segment_logo/YOLO-World/
├── pretrained_models/
│   └── yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth
└── configs/
    └── pretrained_models/  # ← Chú ý: CLIP model nằm ở đây!
        └── open-ai-clip-vit-base-patch32/
            ├── config.json
            ├── pytorch_model.bin
            ├── preprocessor_config.json
            ├── tokenizer_config.json
            ├── tokenizer.json
            ├── vocab.json
            └── merges.txt
```

**Giải thích đường dẫn:**

- `load_from = 'pretrained_models/...'` → Từ thư mục gốc YOLO-World
- `text_model_name = '../pretrained_models/open-ai-clip-vit-base-patch32'` → Từ thư mục configs/ lên 1 cấp rồi vào pretrained_models

## ✅ Kiểm tra sau khi upload

```bash
# Kiểm tra files đã upload đúng chưa
ls -la /home/quangnh58/Segment_logo/YOLO-World/pretrained_models/
ls -la /home/quangnh58/Segment_logo/YOLO-World/pretrained_models/open-ai-clip-vit-base-patch32/

# Kiểm tra kích thước files
du -sh /home/quangnh58/Segment_logo/YOLO-World/pretrained_models/*
```

**⚠️ CHÍNH XÁC HÓA LỆNH UPLOAD:**

```bash
# Upload đúng vị trí theo config
scp yolo_world_models/pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth \
    username@server:/home/quangnh58/Segment_logo/YOLO-World/pretrained_models/

scp -r yolo_world_models/pretrained_models/open-ai-clip-vit-base-patch32/ \
    username@server:/home/quangnh58/Segment_logo/YOLO-World/pretrained_models/
```

## 🎯 Chạy Image Prompt Demo

Sau khi upload xong, bạn có thể chạy:

```bash
cd /home/quangnh58/Segment_logo/YOLO-World

# Test với image demo
python demo/image_demo.py demo/sample_images/bus.jpg \
    configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py \
    pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth \
    --text "person,bus,car" \
    --output-dir outputs/
```

## 💡 Tips

1. **Upload song song**: Có thể upload nhiều files cùng lúc để tiết kiệm thời gian
2. **Kiểm tra checksum**: Sau khi upload, so sánh kích thước files để đảm bảo không bị lỗi
3. **Sử dụng screen/tmux**: Cho các lệnh upload lâu để tránh bị ngắt kết nối
4. **Backup**: Giữ lại files đã tải trên máy cá nhân để tái sử dụng

## ⚠️ Lưu ý

- Đảm bảo có đủ dung lượng trống (~3GB) trên cả máy cá nhân và server
- Kết nối mạng ổn định khi tải files lớn
- Có thể mất 30-60 phút để tải và upload tùy vào tốc độ mạng
