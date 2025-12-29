#!/usr/bin/env python3
"""
合并所有批次的第二次处理后的 JSON 数据
扫描 data/second_process/ 下所有日期子目录，合并成统一的训练数据
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import random


def find_batch_json_files(second_process_dir: Path) -> List[Path]:
    """
    查找所有批次的 JSON 文件
    
    Args:
        second_process_dir: second_process 目录路径
        
    Returns:
        所有找到的 JSON 文件路径列表
    """
    json_files = []
    
    # 遍历所有子目录
    for subdir in sorted(second_process_dir.iterdir()):
        # 跳过 training 目录（这是输出目录）
        if not subdir.is_dir() or subdir.name == 'training':
            continue
        
        # 查找该批次目录下的 JSON 文件
        for json_file in subdir.glob('*.json'):
            json_files.append(json_file)
            print(f"  发现批次文件: {json_file.relative_to(second_process_dir.parent)}")
    
    return json_files


def load_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    加载单个 JSON 文件
    
    Args:
        file_path: JSON 文件路径
        
    Returns:
        对话列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 确保数据是列表格式
        if not isinstance(data, list):
            print(f"  警告: {file_path} 不是列表格式，跳过")
            return []
            
        return data
    except Exception as e:
        print(f"  错误: 无法读取 {file_path}: {e}")
        return []


def merge_all_batches(second_process_dir: Path, output_file: Path) -> int:
    """
    合并所有批次的 JSON 数据
    
    Args:
        second_process_dir: second_process 目录路径
        output_file: 输出文件路径
        
    Returns:
        合并后的对话总数
    """
    print("开始扫描批次文件...")
    json_files = find_batch_json_files(second_process_dir)
    
    if not json_files:
        print("错误: 没有找到任何批次文件")
        return 0
    
    print(f"\n找到 {len(json_files)} 个批次文件")
    print("\n开始合并数据...")
    
    all_conversations = []
    
    for json_file in json_files:
        conversations = load_json_file(json_file)
        if conversations:
            all_conversations.extend(conversations)
            print(f"  ✓ {json_file.name}: {len(conversations)} 条对话")
    
    # 打乱列表
    random.shuffle(all_conversations)
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入合并后的数据
    print(f"\n写入合并后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    return len(all_conversations)


def main():
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 定义路径
    second_process_dir = project_root / 'data' / 'second_process'
    output_file = second_process_dir / 'training' / 'training_all.json'
    
    print("=" * 60)
    print("合并所有批次的训练数据")
    print("=" * 60)
    print(f"扫描目录: {second_process_dir}")
    print(f"输出文件: {output_file}")
    print()
    
    # 检查目录是否存在
    if not second_process_dir.exists():
        print(f"错误: 目录不存在: {second_process_dir}")
        sys.exit(1)
    
    # 合并数据
    total_conversations = merge_all_batches(second_process_dir, output_file)
    
    if total_conversations == 0:
        print("\n错误: 没有合并任何数据")
        sys.exit(1)
    
    # 显示统计信息
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    print()
    print("=" * 60)
    print("合并完成！")
    print("=" * 60)
    print(f"总对话数: {total_conversations}")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {file_size:.2f} MB")
    print("=" * 60)


if __name__ == '__main__':
    main()
