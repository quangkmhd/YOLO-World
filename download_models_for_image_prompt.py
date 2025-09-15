#!/usr/bin/env python3
"""
Script để tải các model files cần thiết cho YOLO-World Image Prompt Demo
Chạy script này trên máy cá nhân có kết nối internet để tải files từ HuggingFace
"""

import os
import requests
from pathlib import Path
import hashlib
from tqdm import tqdm

def download_file(url, local_path, chunk_size=8192):
    """Download file với progress bar"""
    print(f"Downloading: {url}")
    print(f"To: {local_path}")
    
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Progress") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"✅ Downloaded: {local_path}")
    print()

def main():
    print("🚀 YOLO-World Image Prompt Demo - Model Downloader")
    print("=" * 60)
    
    # Tạo thư mục gốc
    base_dir = Path("yolo_world_models")
    base_dir.mkdir(exist_ok=True)
    
    # Danh sách files cần tải
    downloads = [
        {
            "name": "YOLO-World-L Stage1 Model (FIXED URL)",
            "url": "https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth",
            "path": base_dir / "pretrained_models" / "l_stage1-7d280586.pth",
            "size": "~1.4GB"
        },
        {
            "name": "OpenAI CLIP ViT-B/32 - Config",
            "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json",
            "path": base_dir / "pretrained_models" / "open-ai-clip-vit-base-patch32" / "config.json",
            "size": "~1KB"
        },
        {
            "name": "OpenAI CLIP ViT-B/32 - Model",
            "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin",
            "path": base_dir / "pretrained_models" / "open-ai-clip-vit-base-patch32" / "pytorch_model.bin",
            "size": "~600MB"
        },
        {
            "name": "OpenAI CLIP ViT-B/32 - Preprocessor Config",
            "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json",
            "path": base_dir / "pretrained_models" / "open-ai-clip-vit-base-patch32" / "preprocessor_config.json",
            "size": "~1KB"
        },
        {
            "name": "OpenAI CLIP ViT-B/32 - Tokenizer Config",
            "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json",
            "path": base_dir / "pretrained_models" / "open-ai-clip-vit-base-patch32" / "tokenizer_config.json",
            "size": "~1KB"
        },
        {
            "name": "OpenAI CLIP ViT-B/32 - Tokenizer",
            "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json",
            "path": base_dir / "pretrained_models" / "open-ai-clip-vit-base-patch32" / "tokenizer.json",
            "size": "~2MB"
        },
        {
            "name": "OpenAI CLIP ViT-B/32 - Vocab",
            "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json",
            "path": base_dir / "pretrained_models" / "open-ai-clip-vit-base-patch32" / "vocab.json",
            "size": "~1MB"
        },
        {
            "name": "OpenAI CLIP ViT-B/32 - Merges",
            "url": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt",
            "path": base_dir / "pretrained_models" / "open-ai-clip-vit-base-patch32" / "merges.txt",
            "size": "~500KB"
        }
    ]
    
    print(f"📋 Sẽ tải {len(downloads)} files:")
    total_size = 0
    for item in downloads:
        print(f"  - {item['name']} ({item['size']})")
        if "GB" in item['size']:
            total_size += float(item['size'].replace('~', '').replace('GB', '')) * 1024
        elif "MB" in item['size']:
            total_size += float(item['size'].replace('~', '').replace('MB', ''))
    
    print(f"\n📊 Tổng dung lượng ước tính: ~{total_size:.1f}MB")
    print("\n⚠️  Lưu ý: Cần kết nối internet ổn định để tải các file lớn")
    
    confirm = input("\n❓ Bạn có muốn tiếp tục? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Hủy tải xuống")
        return
    
    print("\n🔄 Bắt đầu tải xuống...")
    print("=" * 60)
    
    success_count = 0
    failed_downloads = []
    
    for i, item in enumerate(downloads, 1):
        try:
            print(f"\n[{i}/{len(downloads)}] {item['name']}")
            
            # Kiểm tra file đã tồn tại chưa
            if item['path'].exists():
                print(f"⏭️  File đã tồn tại: {item['path']}")
                success_count += 1
                continue
            
            download_file(item['url'], item['path'])
            success_count += 1
            
        except Exception as e:
            print(f"❌ Lỗi tải {item['name']}: {e}")
            failed_downloads.append(item['name'])
    
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ TẢI XUỐNG:")
    print(f"✅ Thành công: {success_count}/{len(downloads)} files")
    
    if failed_downloads:
        print(f"❌ Thất bại: {len(failed_downloads)} files")
        for name in failed_downloads:
            print(f"  - {name}")
    
    if success_count == len(downloads):
        print("\n🎉 Tải xuống hoàn tất!")
        print(f"📁 Thư mục chứa files: {base_dir.absolute()}")
        print("\n📋 HƯỚNG DẪN TIẾP THEO:")
        print("1. Nén thư mục 'yolo_world_models' thành file .tar.gz hoặc .zip")
        print("2. Upload lên server qua SCP/SFTP:")
        print("   scp -r yolo_world_models/ user@server:/path/to/YOLO-World/")
        print("3. Trên server, di chuyển files đến đúng vị trí:")
        print("   mv yolo_world_models/pretrained_models/* /path/to/YOLO-World/pretrained_models/")
        
        # Tạo script upload
        create_upload_script(base_dir)
    else:
        print("\n⚠️  Một số files không tải được. Hãy kiểm tra kết nối mạng và thử lại.")

def create_upload_script(base_dir):
    """Tạo script để upload lên server"""
    script_content = f"""#!/bin/bash
# Script upload models lên server YOLO-World

# Cấu hình server (sửa theo thông tin của bạn)
SERVER_USER="your_username"
SERVER_HOST="your_server_ip"
SERVER_PATH="/home/quangnh58/Segment_logo/YOLO-World"

echo "🚀 Uploading YOLO-World models to server..."

# Tạo thư mục trên server
ssh $SERVER_USER@$SERVER_HOST "mkdir -p $SERVER_PATH/pretrained_models"

# Upload pretrained models
echo "📤 Uploading pretrained models..."
scp -r {base_dir}/pretrained_models/* $SERVER_USER@$SERVER_HOST:$SERVER_PATH/pretrained_models/

# Kiểm tra upload
echo "✅ Verifying upload..."
ssh $SERVER_USER@$SERVER_HOST "ls -la $SERVER_PATH/pretrained_models/"

echo "🎉 Upload completed!"
echo "Bây giờ bạn có thể chạy image prompt demo trên server"
"""
    
    script_path = base_dir / "upload_to_server.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"\n📝 Đã tạo script upload: {script_path}")
    print("   Sửa thông tin server trong script trước khi chạy!")

if __name__ == "__main__":
    main()
