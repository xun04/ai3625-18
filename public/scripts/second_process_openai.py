#!/usr/bin/env python3
"""
第二步处理脚本：过滤和筛选训练数据
输入：first_process 处理后的 JSON 数据
输出：经过筛选后的 JSON 数据
"""

import json
import argparse
import os
from typing import List, Dict, Any


def filter_by_message_count(data: List[Dict[str, Any]], min_messages: int = 5) -> List[Dict[str, Any]]:
    """
    过滤掉 messages 数量小于指定阈值的样本
    
    Args:
        data: 输入数据列表
        min_messages: 最小 messages 数量阈值
    
    Returns:
        过滤后的数据列表
    """
    filtered_data = []
    removed_count = 0
    
    for sample in data:
        if len(sample.get('messages', [])) >= min_messages:
            filtered_data.append(sample)
        else:
            removed_count += 1
    
    print(f"过滤规则: messages 数量 >= {min_messages}")
    print(f"原始样本数: {len(data)}")
    print(f"保留样本数: {len(filtered_data)}")
    print(f"移除样本数: {removed_count}")
    
    return filtered_data


def merge_consecutive_assistant(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并连续的普通 assistant 消息（非 tool call）
    
    Args:
        data: 输入数据列表
    
    Returns:
        处理后的数据列表
    """
    processed_data = []
    merge_count = 0
    
    for sample in data:
        messages = sample.get('messages', [])
        merged_messages = []
        i = 0
        
        while i < len(messages):
            current_msg = messages[i]
            
            # 如果是普通 assistant 消息（没有 tool_calls）
            if (current_msg.get('role') == 'assistant' and 
                'tool_calls' not in current_msg):
                
                # 收集连续的普通 assistant 消息
                merged_content = current_msg.get('content', '')
                j = i + 1
                
                while (j < len(messages) and 
                       messages[j].get('role') == 'assistant' and 
                       'tool_calls' not in messages[j]):
                    
                    merged_content += '\n' + messages[j].get('content', '')
                    j += 1
                
                # 创建合并后的消息
                merged_msg = {
                    'role': 'assistant',
                    'content': merged_content
                }
                merged_messages.append(merged_msg)
                
                # 如果合并了多个消息，计数
                if j - i > 1:
                    merge_count += 1
                
                i = j
            else:
                # 其他类型的消息直接添加
                merged_messages.append(current_msg)
                i += 1
        
        # 更新样本的 messages
        processed_sample = sample.copy()
        processed_sample['messages'] = merged_messages
        processed_data.append(processed_sample)
    
    print(f"处理规则: 合并连续的普通 assistant 消息")
    print(f"原始样本数: {len(data)}")
    print(f"处理样本数: {len(processed_data)}")
    print(f"合并操作数: {merge_count}")
    
    return processed_data


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='第二步处理：过滤训练数据')
    parser.add_argument('--input', '-i', required=True, help='输入的 JSON 文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出的 JSON 文件路径')
    parser.add_argument('--min_messages', '-m', type=int, default=5, 
                        help='最小 messages 数量阈值 (默认: 5)')
    
    args = parser.parse_args()
    
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"最小 messages 数量: {args.min_messages}")
    print("-" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 读取数据
    print("正在读取输入文件...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 第一步筛选：过滤 messages 数量
    print("\n第一步筛选：过滤 messages 数量")
    filtered_data = filter_by_message_count(data, args.min_messages)
    
    # 第二步处理：合并连续 assistant 消息
    print("\n第二步处理：合并连续 assistant 消息")
    filtered_data = merge_consecutive_assistant(filtered_data)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果
    print(f"\n正在保存到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print("处理完成！")
    return 0


if __name__ == "__main__":
    exit(main())
