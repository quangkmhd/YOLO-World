#!/bin/bash

# YOLO-World Setup Script for CUDA 12.8
# Author: AI Assistant
# Date: $(date)

set -e  # Exit on any error

echo "🚀 YOLO-World Setup Script for CUDA 12.8"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running in conda environment
check_conda() {
    if [[ "$CONDA_DEFAULT_ENV" != "" ]]; then
        print_status "Detected conda environment: $CONDA_DEFAULT_ENV"
        return 0
    else
        print_warning "No conda environment detected. Please activate conda environment first:"
        echo "conda create -n yoloworld python=3.8"
        echo "conda activate yoloworld"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Check CUDA availability
check_cuda() {
    print_step "Checking CUDA installation..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
        print_status "CUDA driver detected"
    else
        print_error "nvidia-smi not found. Please install NVIDIA drivers and CUDA toolkit."
        exit 1
    fi
    
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_status "NVCC version: $CUDA_VERSION"
    else
        print_warning "nvcc not found. CUDA toolkit may not be properly installed."
    fi
}

# Install PyTorch with CUDA 12.1 (compatible with 12.8)
install_pytorch() {
    print_step "Installing PyTorch with CUDA 12.1 support..."
    
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    
    # Test PyTorch CUDA
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name()}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available!')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "PyTorch with CUDA installed successfully!"
    else
        print_error "PyTorch CUDA installation failed!"
        exit 1
    fi
}

# Install basic dependencies
install_dependencies() {
    print_step "Installing basic dependencies..."
    
    # Install wheel and setuptools
    pip install wheel setuptools
    
    # Install openmim and mmcv
    print_status "Installing OpenMMLab packages..."
    pip install openmim
    mim install mmcv==2.1.0
    
    # Install requirements
    if [ -f "requirements/basic_requirements.txt" ]; then
        print_status "Installing basic requirements..."
        pip install -r requirements/basic_requirements.txt
    else
        print_warning "basic_requirements.txt not found, installing manually..."
        pip install opencv-python==4.9.0.80
        pip install mmdet==3.0.0
        pip install mmengine==0.10.3
        pip install mmyolo==0.6.0
        pip install timm==0.6.13
        pip install transformers==4.36.2
        pip install albumentations
    fi
    
    # Install demo requirements
    if [ -f "requirements/demo_requirements.txt" ]; then
        print_status "Installing demo requirements..."
        pip install -r requirements/demo_requirements.txt
    else
        pip install gradio==4.16.0 supervision
    fi
}

# Install YOLO-World
install_yoloworld() {
    print_step "Installing YOLO-World..."
    
    # Install in development mode
    pip install -e .
    
    # Test installation
    python -c "
try:
    import yolo_world
    print('YOLO-World imported successfully!')
except ImportError as e:
    print(f'Failed to import YOLO-World: {e}')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "YOLO-World installed successfully!"
    else
        print_error "YOLO-World installation failed!"
        exit 1
    fi
}

# Download pretrained models
download_models() {
    print_step "Setting up model weights directory..."
    
    mkdir -p weights
    cd weights
    
    echo "Available models:"
    echo "1. YOLO-World-S (Small, fastest)"
    echo "2. YOLO-World-M (Medium, balanced)"
    echo "3. YOLO-World-L (Large, most accurate)"
    echo "4. All models"
    
    read -p "Which model would you like to download? (1-4): " model_choice
    
    case $model_choice in
        1)
            print_status "Downloading YOLO-World-S..."
            wget -O s_stage1.pth https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage1-7e1e5299.pth
            ;;
        2)
            print_status "Downloading YOLO-World-M..."
            wget -O m_stage1.pth https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage1-7e1e5299.pth
            ;;
        3)
            print_status "Downloading YOLO-World-L..."
            wget -O l_stage1.pth https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth
            ;;
        4)
            print_status "Downloading all models..."
            wget -O s_stage1.pth https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/s_stage1-7e1e5299.pth &
            wget -O m_stage1.pth https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/m_stage1-7e1e5299.pth &
            wget -O l_stage1.pth https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth &
            wait
            ;;
        *)
            print_warning "Invalid choice. Skipping model download."
            ;;
    esac
    
    cd ..
    print_status "Model download completed!"
}

# Run basic test
run_test() {
    print_step "Running basic functionality test..."
    
    # Test with sample image if available
    if [ -f "demo/sample_images/bus.jpg" ]; then
        print_status "Testing with sample image..."
        
        # Find available model
        MODEL_PATH=""
        CONFIG_PATH="configs/pretrain/yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        
        if [ -f "weights/l_stage1.pth" ]; then
            MODEL_PATH="weights/l_stage1.pth"
        elif [ -f "weights/m_stage1.pth" ]; then
            MODEL_PATH="weights/m_stage1.pth"
            CONFIG_PATH="configs/pretrain/yolo_world_v2_m_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        elif [ -f "weights/s_stage1.pth" ]; then
            MODEL_PATH="weights/s_stage1.pth"
            CONFIG_PATH="configs/pretrain/yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py"
        fi
        
        if [ -n "$MODEL_PATH" ] && [ -f "$CONFIG_PATH" ]; then
            mkdir -p outputs
            python demo/image_demo.py demo/sample_images/bus.jpg \
                $CONFIG_PATH \
                $MODEL_PATH \
                --text "person,bus,car,bicycle" \
                --topk 100 \
                --threshold 0.005 \
                --output-dir outputs/
            
            if [ $? -eq 0 ]; then
                print_status "Test completed successfully! Check outputs/ directory for results."
            else
                print_warning "Test completed with some issues."
            fi
        else
            print_warning "No model or config found for testing."
        fi
    else
        print_warning "Sample image not found. Skipping image test."
    fi
}

# Main installation process
main() {
    print_status "Starting YOLO-World installation process..."
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ] || [ ! -d "yolo_world" ]; then
        print_error "Please run this script from the YOLO-World root directory!"
        exit 1
    fi
    
    check_conda
    check_cuda
    install_pytorch
    install_dependencies
    install_yoloworld
    
    read -p "Do you want to download pretrained models? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        download_models
    fi
    
    read -p "Do you want to run a basic test? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        run_test
    fi
    
    print_status "🎉 YOLO-World installation completed!"
    echo
    echo "Next steps:"
    echo "1. Try the Gradio demo: python demo/gradio_demo.py"
    echo "2. Run image inference: python demo/image_demo.py [image_path] [config] [weights] --text 'object1,object2'"
    echo "3. Check the documentation: https://github.com/AILab-CVC/YOLO-World"
    echo
    echo "For troubleshooting, check SETUP_GUIDE_VI.md"
}

# Run main function
main "$@"
