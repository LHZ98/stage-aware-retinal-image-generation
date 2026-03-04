#!/usr/bin/env python3
"""
下载ROP数据集脚本
Mendeley数据集通常需要登录后下载，此脚本提供下载指南和可能的自动化下载
"""

import os
import requests
from pathlib import Path
import zipfile
import tarfile

def download_rop_dataset(output_dir: str = "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/ROP"):
    """
    尝试下载ROP数据集
    
    注意：Mendeley数据集通常需要：
    1. 登录Mendeley账户
    2. 同意使用条款
    3. 手动下载或使用API
    
    如果自动下载失败，请手动下载
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ROP数据集下载")
    print("=" * 60)
    print(f"目标目录: {output_dir}")
    print()
    
    # Mendeley数据集URL
    dataset_url = "https://data.mendeley.com/datasets/prcy36j53v/2"
    
    print("方法1: 尝试直接下载...")
    try:
        # 尝试获取下载页面
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(dataset_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("✓ 可以访问数据集页面")
            print("⚠️  但Mendeley通常需要登录才能下载")
        else:
            print(f"✗ 无法访问: HTTP {response.status_code}")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
    
    print()
    print("=" * 60)
    print("手动下载指南")
    print("=" * 60)
    print("由于Mendeley数据集需要登录，请按以下步骤手动下载：")
    print()
    print("1. 访问数据集页面:")
    print(f"   {dataset_url}")
    print()
    print("2. 登录Mendeley账户（如果没有，需要先注册）")
    print()
    print("3. 点击 'Download All' 按钮下载数据集")
    print()
    print("4. 下载完成后，将压缩文件放到以下目录:")
    print(f"   {output_dir}")
    print()
    print("5. 运行解压脚本:")
    print(f"   python {__file__} --extract")
    print()
    print("=" * 60)
    
    # 检查是否已有下载的文件
    zip_files = list(output_path.glob("*.zip"))
    tar_files = list(output_path.glob("*.tar*"))
    
    if zip_files or tar_files:
        print("\n发现压缩文件:")
        for f in zip_files + tar_files:
            print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        print("\n运行解压:")
        print(f"  python {__file__} --extract")
    
    return output_dir


def extract_dataset(data_dir: str = "/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/ROP"):
    """
    解压ROP数据集
    """
    data_path = Path(data_dir)
    
    print("=" * 60)
    print("解压ROP数据集")
    print("=" * 60)
    
    # 查找压缩文件
    zip_files = list(data_path.glob("*.zip"))
    tar_files = list(data_path.glob("*.tar*"))
    
    if not zip_files and not tar_files:
        print("✗ 未找到压缩文件")
        print(f"请将下载的压缩文件放到: {data_dir}")
        return
    
    for archive in zip_files + tar_files:
        print(f"\n解压: {archive.name}")
        
        try:
            if archive.suffix == '.zip':
                with zipfile.ZipFile(archive, 'r') as zip_ref:
                    zip_ref.extractall(data_path)
                print(f"✓ 解压完成")
            elif archive.suffix in ['.tar', '.gz', '.bz2']:
                with tarfile.open(archive, 'r:*') as tar_ref:
                    tar_ref.extractall(data_path)
                print(f"✓ 解压完成")
            else:
                print(f"✗ 不支持的压缩格式: {archive.suffix}")
        except Exception as e:
            print(f"✗ 解压失败: {e}")
    
    # 检查解压后的目录结构
    print("\n检查目录结构...")
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    files = [f for f in data_path.iterdir() if f.is_file() and f.suffix in ['.png', '.jpg', '.tif']]
    
    if subdirs:
        print(f"发现 {len(subdirs)} 个子目录:")
        for d in subdirs[:5]:
            print(f"  - {d.name}")
        if len(subdirs) > 5:
            print(f"  ... 还有 {len(subdirs) - 5} 个目录")
    
    if files:
        print(f"发现 {len(files)} 个图像文件")
    
    print("\n✓ 解压完成！")
    print(f"数据位置: {data_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载或解压ROP数据集')
    parser.add_argument('--extract', action='store_true',
                       help='解压已下载的压缩文件')
    parser.add_argument('--output_dir', type=str,
                       default="/home/yuganlab/Desktop/Haozhe/MICCAI2026/datasets/masks/ROP",
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.extract:
        extract_dataset(args.output_dir)
    else:
        download_rop_dataset(args.output_dir)

