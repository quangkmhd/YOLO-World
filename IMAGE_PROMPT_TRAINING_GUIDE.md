# Hướng dẫn Training Image Prompt YOLO-World

## 🎯 Tổng quan Image Prompt Training

Image Prompt Training là phương pháp training YOLO-World để có thể detect objects bằng cách sử dụng cả **text prompts** và **image prompts** (ảnh mẫu của object cần detect).

## 📋 Quy trình Training từng bước

### Bước 1: Chuẩn bị môi trường và dependencies
```bash
# Setup YOLO-World environment
conda create -n yoloworld python=3.8
conda activate yoloworld
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install openmim
mim install mmcv==2.1.0
pip install -e .
```

**Files cần thiết:**
- `setup_yoloworld.sh` - Script tự động setup
- `requirements/basic_requirements.txt` - Dependencies cơ bản
- `pyproject.toml` - Package configuration

### Bước 2: Tải Pre-trained Models
```bash
# Tải YOLO-World base model (đã pre-train)
wget -O pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth \
  "https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth"

# Tải CLIP model cho text/image encoding
mkdir -p pretrained_models/open-ai-clip-vit-base-patch32
# ... (download CLIP files)
```

**Files cần thiết:**
- `yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth` - Base YOLO-World model
- `pretrained_models/open-ai-clip-vit-base-patch32/` - CLIP model files
- `download_models_for_image_prompt.py` - Script tải models

### Bước 3: Chuẩn bị Dataset
```bash
# Cấu trúc dataset cho image prompt training
data/
├── coco/
│   ├── train2017/          # Training images
│   ├── val2017/            # Validation images
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
└── texts/
    └── coco_class_texts.json  # Text descriptions cho classes
```

**Files cần thiết:**
- `data/texts/coco_class_texts.json` - Text prompts cho các class
- Dataset images và annotations
- Custom dataset converter (nếu dùng custom data)

### Bước 4: Cấu hình Training Config
**File chính:** `configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py`

**Các thành phần quan trọng:**
```python
# Base model để load weights
load_from = 'pretrained_models/yolo_world_l_clip_t2i_bn_2e-3adamw_32xb16-100e_obj365v1_goldg_cc3mlite_train-ca93cd1f.pth'

# CLIP model cho text/image encoding (FROZEN)
text_model_name = '../pretrained_models/open-ai-clip-vit-base-patch32'

# Model architecture
model = dict(
    type='YOLOWorldImageDetector',  # ← Image prompt detector
    vision_model=text_model_name,   # ← Sử dụng CLIP vision
    backbone=dict(
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            frozen_modules=['all']  # ← CLIP bị đóng băng
        )
    ),
    neck=dict(freeze_all=True),     # ← Neck bị đóng băng
    bbox_head=dict(freeze_all=True) # ← Head bị đóng băng
)
```

### Bước 5: Chạy Training
```bash
# Single GPU training
python tools/train.py configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py

# Multi-GPU training (8 GPUs)
./tools/dist_train.sh configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py 8 --amp

# Custom work directory
python tools/train.py configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py \
    --work-dir work_dirs/image_prompt_training
```

**Files sử dụng:**
- `tools/train.py` - Main training script
- `tools/dist_train.sh` - Distributed training script
- Config file - Training configuration

### Bước 6: Monitoring và Evaluation
```bash
# Theo dõi training progress
tensorboard --logdir work_dirs/image_prompt_training

# Evaluation trên validation set
python tools/test.py configs/image_prompts/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py \
    work_dirs/image_prompt_training/latest.pth \
    --out results.pkl
```

**Files được tạo:**
- `work_dirs/image_prompt_training/` - Training outputs
- `latest.pth`, `best.pth` - Model checkpoints
- `scalars.json` - Training metrics
- `results.pkl` - Evaluation results

## 🔧 Chi tiết từng thành phần

### 1. **YOLOWorldImageDetector**
- **Mục đích:** Detector hỗ trợ cả text và image prompts
- **Input:** Images + text descriptions + image examples
- **Output:** Bounding boxes + class predictions

### 2. **CLIP Model (Frozen)**
- **Mục đích:** Extract semantic features từ text và images
- **Vai trò:** Feature extractor cố định, không được training
- **Lý do freeze:** Giữ nguyên pre-trained semantic knowledge

### 3. **MultiModalYOLOBackbone**
- **Mục đích:** Kết hợp CLIP features với YOLO detection
- **Components:**
  - `image_model`: YOLO backbone cho image features
  - `text_model`: CLIP text encoder (frozen)
  - `frozen_stages`: Freeze một số layer của image model

### 4. **Training Strategy**
```python
# Chỉ train các layer alignment
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # CLIP lr rất nhỏ
            'logit_scale': dict(weight_decay=0.0),
            'embeddings': dict(weight_decay=0.0)
        }
    )
)
```

## 📊 Luồng dữ liệu trong Training

```
Input Images → YOLO Backbone → Image Features
                    ↓
Text Prompts → CLIP Text Encoder → Text Features
                    ↓
Image Examples → CLIP Vision Encoder → Visual Features
                    ↓
            Multimodal Fusion
                    ↓
            Detection Head → Predictions
                    ↓
            Loss Calculation → Backprop
```

## 🎯 Kết quả Training

**Model sau training có thể:**
1. Detect objects bằng text descriptions
2. Detect objects bằng image examples
3. Zero-shot detection trên classes mới
4. Few-shot learning với ít examples

**Files output:**
- `best.pth` - Model weights tốt nhất
- `config.py` - Training configuration
- `scalars.json` - Training metrics
- `log.txt` - Training logs

## 💡 Tips Training hiệu quả

1. **Memory Management:**
   - Sử dụng `--amp` cho mixed precision
   - Giảm batch size nếu OOM
   - Freeze nhiều layers để tiết kiệm memory

2. **Data Preparation:**
   - Chuẩn bị text descriptions chất lượng cao
   - Sử dụng diverse image examples
   - Balance dataset giữa các classes

3. **Hyperparameter Tuning:**
   - Learning rate thấp cho frozen components
   - Warmup scheduler cho stable training
   - Early stopping để tránh overfitting

## 🚀 Sử dụng Model sau Training

```python
# Load trained model
from yolo_world import YOLOWorldImageDetector
model = YOLOWorldImageDetector.from_pretrained('work_dirs/image_prompt_training/best.pth')

# Inference với text prompt
results = model.predict(image, text_prompts=["person", "car"])

# Inference với image prompt
results = model.predict(image, image_prompts=[example_person_img, example_car_img])
```

Quy trình này cho phép training YOLO-World để có khả năng detect objects bằng cả text và image prompts, tận dụng sức mạnh của CLIP để hiểu semantic content.
