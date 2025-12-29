import os
import json
import pdb
import argparse
import pandas as pd
from copy import deepcopy
from transformers import AutoTokenizer

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将OpenAI格式的JSON数据转换为Parquet格式')
    parser.add_argument('--tokenizer_path', '-t', required=True, help='Tokenizer模型路径')
    parser.add_argument('--input_json', '-i', required=True, help='输入的OpenAI JSON文件路径')
    parser.add_argument('--output_parquet', '-o', required=True, help='输出的Parquet文件路径')
    parser.add_argument('--max_tokens', '-m', type=int, default=128000, help='最大token数限制 (默认: 128000)')
    parser.add_argument('--duplicate_times', '-d', type=int, default=10, help='数据复制倍数 (默认: 10)')
    
    args = parser.parse_args()
    
    print(f"Tokenizer路径: {args.tokenizer_path}")
    print(f"输入JSON文件: {args.input_json}")
    print(f"输出Parquet文件: {args.output_parquet}")
    print(f"最大token数: {args.max_tokens}")
    print(f"数据复制倍数: {args.duplicate_times}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_json):
        print(f"错误: 输入JSON文件不存在: {args.input_json}")
        return 1
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(tokenizer.chat_template)
    
    # 读取数据
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))

    def fix_tool_calls(messages):
        for m in messages:
            if m.get("role") == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except Exception:
                            func["arguments"] = {}
            elif m.get("role") == "tool" and isinstance(m.get("content"), str):
                # 如果 tool 的 content 也是字符串列表，也可以按需处理
                pass
        return messages

    save_data = []
    key_type = {}
    for sample in data:
        conflict = False
        full_prompt = tokenizer.apply_chat_template(
            fix_tool_calls(sample["messages"]),
            tools=sample.get("tools", []),
            tokenize=False,
            add_generation_prompt=False
        )
        for idx in range(len(sample['messages'])):
            if "tool_calls" in sample["messages"][idx]:
                for call in sample["messages"][idx]["tool_calls"]:
                    call_new = deepcopy(call)
                    for key in call["function"]["arguments"].keys():
                        if key in key_type.keys():
                            if type(call["function"]["arguments"][key]) != key_type[key]:
                                print(f"{key} | {type(call['function']['arguments'][key])} v.s. key_type={key_type[key]}")
                                # call_new["function"]["arguments"].pop(key)
                                conflict = True
                        else:
                            key_type[key] = type(call["function"]["arguments"][key])
                    # sample["messages"][idx]["tool_calls"] = call_new
                # sample["messages"][idx]["tool_calls"] = json.dumps(sample["messages"][idx]["tool_calls"])
        # if len(tokenizer.encode(full_prompt, add_special_tokens=False)) > 128000 or conflict:
        if len(tokenizer.encode(full_prompt, add_special_tokens=False)) > args.max_tokens or conflict:
            continue
        # with open("/inspire/hdd/project/qproject-fundationmodel/public/sunjie/sft/dataset/train_trajectory/full_promt.txt", "a", encoding="utf-8") as f:
        #     f.write(full_prompt + "\n")
        save_data.append(sample)

    # with open(f"/inspire/hdd/project/qproject-fundationmodel/public/sunjie/sft/dataset/train_trajectory_trajectory/train_250827_{len(save_data)}sample-128k.json", "w", encoding="utf-8") as f:
    #     json.dump(save_data, f, ensure_ascii=False, indent=4)

    save_data = save_data * args.duplicate_times
    print(len(save_data))
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(save_data).to_parquet(args.output_parquet)
    
    return 0

if __name__ == "__main__":
    exit(main())