#!/usr/bin/env python3
"""
批量将HEIC图片转换为JPG格式，并更新transforms.json
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("警告: pillow-heif未安装，请运行: pip install pillow-heif")
    print("如果已安装但仍报错，可能需要安装系统依赖: libheif-dev (Ubuntu/Debian)")
    exit(1)


def convert_and_sort(input_dir, transforms_json, output_dir):
    """按照transforms.json顺序转换图片并更新file_path"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    transforms_path = Path(transforms_json)
    
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return None
    
    if not transforms_path.exists():
        print(f"错误: transforms.json不存在: {transforms_json}")
        return None
    
    # 读取transforms.json
    with open(transforms_path, 'r', encoding='utf-8') as f:
        transforms_data = json.load(f)
    
    frames = transforms_data.get('frames', [])
    if not frames:
        print("错误: transforms.json中没有frames数据")
        return None
    
    print(f"找到 {len(frames)} 个frames，开始处理...")
    
    # 创建输出目录
    output_image_dir = output_path / 'image'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个frame
    success_count = 0
    for idx, frame in enumerate(frames, start=1):
        try:
            # 获取原始file_path
            old_file_path = frame.get('file_path', '')
            if not old_file_path:
                print(f"✗ Frame {idx}: 缺少file_path")
                continue
            
            # 解析原始文件路径
            old_path = Path(old_file_path)
            if old_path.is_absolute():
                source_file = old_path
            else:
                # 相对路径，从输入目录查找
                source_file = input_path / old_path.name
            
            if not source_file.exists():
                print(f"✗ Frame {idx}: 文件不存在 {source_file}")
                continue
            
            # 生成新的文件名
            new_filename = f"{idx:08d}.jpg"
            output_file = output_image_dir / new_filename
            
            # 转换图片
            img = Image.open(source_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output_file, 'JPEG', quality=100)
            
            # 更新file_path
            frame['file_path'] = f"./image/{new_filename}"
            
            print(f"✓ {idx}/{len(frames)}: {source_file.name} -> {new_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Frame {idx}: 处理失败 - {e}")
    
    # 保存更新后的transforms.json
    output_transforms = output_path / 'transforms.json'
    with open(output_transforms, 'w', encoding='utf-8') as f:
        json.dump(transforms_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n完成: 成功处理 {success_count}/{len(frames)} 个文件")
    print(f"输出目录: {output_path}")
    
    return transforms_data


def sample_images(sorted_dir, output_dir, count=100):
    """从排序后的数据中均匀抽取指定数量的图片和transform数据"""
    sorted_path = Path(sorted_dir)
    output_path = Path(output_dir)
    
    if not sorted_path.exists():
        print(f"错误: 排序数据目录不存在: {sorted_dir}")
        return
    
    sorted_transforms = sorted_path / 'transforms.json'
    sorted_image_dir = sorted_path / 'image'
    
    if not sorted_transforms.exists() or not sorted_image_dir.exists():
        print(f"错误: 排序数据不完整")
        return
    
    # 读取排序后的transforms.json
    with open(sorted_transforms, 'r', encoding='utf-8') as f:
        transforms_data = json.load(f)
    
    frames = transforms_data.get('frames', [])
    total_frames = len(frames)
    
    if total_frames < count:
        print(f"警告: 总帧数({total_frames})少于需要抽取的数量({count})")
        count = total_frames
    
    # 计算间隔（从1开始，均匀抽取）
    # 例如：300张选100张，间隔=3，选1,4,7,10,...,298
    if count == 1:
        indices = [0]
    else:
        step = total_frames / count
        indices = [int(i * step) for i in range(count)]
        # 确保索引从0开始（对应图片1）
        indices = [max(0, idx) for idx in indices]
    
    print(f"从 {total_frames} 张图片中均匀抽取 {count} 张...")
    print(f"抽取索引: {indices[:5]}...{indices[-5:]}")
    
    # 创建输出目录
    output_image_dir = output_path / 'image'
    output_image_dir.mkdir(parents=True, exist_ok=True)
    
    # 抽取frames和图片
    sampled_frames = []
    for new_idx, old_idx in enumerate(indices, start=1):
        frame = frames[old_idx].copy()
        old_file_path = frame.get('file_path', '')
        
        # 解析原图片路径
        old_image_name = Path(old_file_path).name
        old_image_path = sorted_image_dir / old_image_name
        
        if not old_image_path.exists():
            print(f"✗ 跳过: 图片不存在 {old_image_name}")
            continue
        
        # 生成新的文件名
        new_filename = f"{new_idx:08d}.jpg"
        new_image_path = output_image_dir / new_filename
        
        # 复制图片
        shutil.copy2(old_image_path, new_image_path)
        
        # 更新file_path
        frame['file_path'] = f"./image/{new_filename}"
        sampled_frames.append(frame)
        
        print(f"✓ {new_idx}/{count}: {old_image_name} -> {new_filename}")
    
    # 更新transforms.json
    transforms_data['frames'] = sampled_frames
    output_transforms = output_path / 'transforms.json'
    with open(output_transforms, 'w', encoding='utf-8') as f:
        json.dump(transforms_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n完成: 抽取 {len(sampled_frames)} 张图片到 {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量转换HEIC图片并更新transforms.json')
    parser.add_argument('--input_dir', help='输入图片目录路径')
    parser.add_argument('--transforms_json', default='transforms.json', help='transforms.json路径')
    parser.add_argument('--output_dir', default='./sort_data', help='输出目录路径')
    parser.add_argument('--sample_count', type=int, default=100, help='抽取的图片数量')
    parser.add_argument('--sample_output', default='./sort_data_100', help='抽取数据输出目录')
    parser.add_argument('--only_sample', action='store_true', help='仅执行抽取步骤（跳过转换）')
    
    args = parser.parse_args()
    
    if not args.only_sample:
        if not args.input_dir:
            print("错误: 需要指定 --input_dir")
            exit(1)
        
        # 执行转换和排序
        convert_and_sort(args.input_dir, args.transforms_json, args.output_dir)
    
    # 执行抽取
    sample_images(args.output_dir, args.sample_output, args.sample_count)
