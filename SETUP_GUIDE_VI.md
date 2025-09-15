# Hướng dẫn Setup YOLO-World từ đầu với CUDA 12.8

## Yêu cầu hệ thống
- Python 3.7 - 3.11
- CUDA 12.8
- GPU với tối thiểu 8GB VRAM (khuyến nghị 16GB+)
- Ubuntu/Linux (khuyến nghị)

## Bước 1: Chuẩn bị môi trường Python

### 1.1 Tạo môi trường ảo (Virtual Environment)
```bash
# Sử dụng conda (khuyến nghị)
conda create -n yoloworld python=3.8
conda activate yoloworld

# Hoặc sử dụng venv
python -m venv yoloworld_env
source yoloworld_env/bin/activate  # Linux/Mac
# yoloworld_env\Scripts\activate  # Windows
```

### 1.2 Kiểm tra CUDA
```bash
nvidia-smi
nvcc --version
```

## Bước 2: Cài đặt PyTorch với CUDA 12.8

```bash
# Cài đặt PyTorch tương thích với CUDA 12.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Kiểm tra PyTorch có nhận diện GPU không
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

## Bước 3: Clone và Setup YOLO-World

### 3.1 Clone repository
```bash
cd /home/quangnh58/Segment_logo/
git clone --recursive https://github.com/AILab-CVC/YOLO-World.git
cd YOLO-World
```

### 3.2 Cài đặt dependencies cơ bản
```bash
# Cài đặt wheel và setuptools
pip install wheel setuptools

# Cài đặt mmcv cho CUDA 12.1 (tương thích với 12.8)
pip install openmim
mim install mmcv==2.1.0

# Hoặc cài đặt trực tiếp
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

### 3.3 Cài đặt các dependencies từ requirements
```bash
# Cài đặt basic requirements
pip install -r requirements/basic_requirements.txt

# Cài đặt demo requirements (nếu muốn chạy demo)
pip install -r requirements/demo_requirements.txt
```

### 3.4 Cài đặt YOLO-World
```bash
# Cài đặt YOLO-World ở chế độ development
pip install -e .
```

## Bước 4: Tải Pre-trained Models

### 4.1 Tạo thư mục weights
```bash
mkdir -p weights
cd weights
```

### 4.2 Tải model weights (chọn 1 trong các model sau)

**YOLO-World-S (Small - nhanh nhất):**
```bash
wget https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage1-7e1e5299.pth
```

**YOLO-World-M (Medium - cân bằng):**
```bash
wget https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage1-7e1e5299.pth
```

**YOLO-World-L (Large - chính xác nhất):**
```bash
wget https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth
```

## Bước 5: Test Installation

### 5.1 Test cơ bản
```bash
cd /home/quangnh58/Segment_logo/YOLO-World

# Test import
python -c "import yolo_world; print('YOLO-World imported successfully!')"

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 5.2 Chạy demo đơn giản
```bash
# Chạy image demo với ảnh mẫu
python demo/image_demo.py demo/sample_images/bus.jpg \
    configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    weights/l_stage1-7d280586.pth \
    --text "person,bus,car" \
    --topk 100 \
    --threshold 0.005 \
    --output-dir outputs/
```

## Bước 6: Chạy Gradio Demo (Tùy chọn)

```bash
# Cài đặt gradio nếu chưa có
pip install gradio==4.16.0

# Chạy gradio demo
python demo/gradio_demo.py
```

## Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory:**
   - Giảm batch size
   - Sử dụng model nhỏ hơn (S thay vì L)
   - Giảm resolution input

2. **Import error mmcv:**
   ```bash
   pip uninstall mmcv mmcv-full mmcv-lite
   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
   ```

3. **Torch version mismatch:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Git submodule issues:**
   ```bash
   git submodule update --init --recursive
   ```

## Các lệnh hữu ích

### Kiểm tra GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Chạy inference trên video:
```bash
python demo/video_demo.py path/to/video.mp4 \
    configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    weights/l_stage1-7d280586.pth \
    --text "person,car,bicycle" \
    --output-dir outputs/
```

### Fine-tuning trên custom dataset:
```bash
# Xem hướng dẫn chi tiết trong docs/finetuning.md
python tools/train.py configs/finetune_coco/yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py
```

## Cấu hình khuyến nghị

- **GPU:** RTX 3080/4080 trở lên (16GB+ VRAM)
- **RAM:** 32GB+
- **Storage:** SSD với ít nhất 50GB trống
- **Python:** 3.8 hoặc 3.9 (ổn định nhất)

## Tài liệu tham khảo

- [Official YOLO-World Repository](https://github.com/AILab-CVC/YOLO-World)
- [MMDetection Documentation](https://mmdetection.readthedocs.io/)
- [MMCV Installation Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
