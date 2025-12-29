#!/usr/bin/env python3
"""
OpenAI 格式数据验证单元测试脚本
用于验证训练数据是否符合 OpenAI 的对话格式要求

Usage:
    python test_openai_format.py --input <path_to_json_file>
    
Example:
    python test_openai_format.py --input ../data/second_process/20251223/training_filtered.json
"""

import json
import argparse
import sys
from typing import List, Dict, Any, Set
from pathlib import Path


class OpenAIFormatValidator:
    """OpenAI 训练数据格式验证器"""
    
    # 允许的角色类型
    VALID_ROLES = {"system", "user", "assistant", "tool"}
    
    # tool_call 必需的字段
    TOOL_CALL_REQUIRED_FIELDS = {"id", "type", "function"}
    FUNCTION_REQUIRED_FIELDS = {"name", "arguments"}
    
    # tool message 必需的字段
    TOOL_MESSAGE_REQUIRED_FIELDS = {"role", "tool_call_id", "content"}
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.stats = {
            "total_conversations": 0,
            "total_messages": 0,
            "role_distribution": {},
            "tool_calls_count": 0,
            "tool_responses_count": 0,
            "avg_messages_per_conversation": 0,
        }
    
    def log_error(self, conv_idx: int, msg: str):
        """记录错误"""
        error_msg = f"[对话 {conv_idx}] {msg}"
        self.errors.append(error_msg)
        if self.verbose:
            print(f"❌ ERROR: {error_msg}")
    
    def log_warning(self, conv_idx: int, msg: str):
        """记录警告"""
        warning_msg = f"[对话 {conv_idx}] {msg}"
        self.warnings.append(warning_msg)
        if self.verbose:
            print(f"⚠️  WARNING: {warning_msg}")
    
    def validate_tool_call(self, conv_idx: int, msg_idx: int, tool_call: Dict[str, Any]) -> bool:
        """验证单个 tool_call 的格式"""
        valid = True
        
        # 检查必需字段
        missing_fields = self.TOOL_CALL_REQUIRED_FIELDS - set(tool_call.keys())
        if missing_fields:
            self.log_error(conv_idx, f"消息 {msg_idx}: tool_call 缺少字段: {missing_fields}")
            valid = False
        
        # 检查 type 必须是 "function"
        if tool_call.get("type") != "function":
            self.log_error(conv_idx, f"消息 {msg_idx}: tool_call.type 必须是 'function', 实际值: {tool_call.get('type')}")
            valid = False
        
        # 检查 function 字段
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            self.log_error(conv_idx, f"消息 {msg_idx}: tool_call.function 必须是对象")
            valid = False
        else:
            missing_func_fields = self.FUNCTION_REQUIRED_FIELDS - set(function.keys())
            if missing_func_fields:
                self.log_error(conv_idx, f"消息 {msg_idx}: tool_call.function 缺少字段: {missing_func_fields}")
                valid = False
            
            # 检查 arguments 必须是字符串或字典
            arguments = function.get("arguments")
            if arguments is not None and not isinstance(arguments, (str, dict)):
                self.log_error(conv_idx, f"消息 {msg_idx}: function.arguments 必须是字符串或字典，实际类型: {type(arguments)}")
                valid = False
        
        return valid
    
    def validate_message(self, conv_idx: int, msg_idx: int, message: Dict[str, Any]) -> bool:
        """验证单条消息格式"""
        valid = True
        
        # 检查必需字段 role
        if "role" not in message:
            self.log_error(conv_idx, f"消息 {msg_idx}: 缺少 'role' 字段")
            return False
        
        role = message["role"]
        
        # 检查 role 是否合法
        if role not in self.VALID_ROLES:
            self.log_error(conv_idx, f"消息 {msg_idx}: 非法的 role 值 '{role}', 必须是 {self.VALID_ROLES} 之一")
            valid = False
        
        # 根据不同 role 检查必需字段
        if role == "tool":
            # tool 消息必须有 tool_call_id 和 content
            missing_fields = self.TOOL_MESSAGE_REQUIRED_FIELDS - set(message.keys())
            if missing_fields:
                self.log_error(conv_idx, f"消息 {msg_idx}: tool 消息缺少字段: {missing_fields}")
                valid = False
        else:
            # 其他角色必须有 content 字段（可以为空字符串）
            if "content" not in message:
                self.log_error(conv_idx, f"消息 {msg_idx}: {role} 消息缺少 'content' 字段")
                valid = False
        
        # 检查 tool_calls（如果存在）
        if "tool_calls" in message:
            if role != "assistant":
                self.log_error(conv_idx, f"消息 {msg_idx}: 只有 assistant 消息可以有 tool_calls")
                valid = False
            
            tool_calls = message["tool_calls"]
            if not isinstance(tool_calls, list):
                self.log_error(conv_idx, f"消息 {msg_idx}: tool_calls 必须是数组")
                valid = False
            else:
                for tc_idx, tool_call in enumerate(tool_calls):
                    if not self.validate_tool_call(conv_idx, msg_idx, tool_call):
                        valid = False
                    else:
                        self.stats["tool_calls_count"] += 1
        
        # 统计角色分布
        self.stats["role_distribution"][role] = self.stats["role_distribution"].get(role, 0) + 1
        
        if role == "tool":
            self.stats["tool_responses_count"] += 1
        
        return valid
    
    def validate_conversation(self, conv_idx: int, conversation: Dict[str, Any]) -> bool:
        """验证单个对话"""
        valid = True
        
        # 检查 messages 字段
        if "messages" not in conversation:
            self.log_error(conv_idx, "缺少 'messages' 字段")
            return False
        
        messages = conversation["messages"]
        if not isinstance(messages, list):
            self.log_error(conv_idx, "'messages' 必须是数组")
            return False
        
        if len(messages) == 0:
            self.log_warning(conv_idx, "对话中没有消息")
            return False
        
        self.stats["total_messages"] += len(messages)
        
        # 检查每条消息
        for msg_idx, message in enumerate(messages):
            if not isinstance(message, dict):
                self.log_error(conv_idx, f"消息 {msg_idx}: 消息必须是对象")
                valid = False
                continue
            
            if not self.validate_message(conv_idx, msg_idx, message):
                valid = False
        
        # 检查对话流程的合理性
        if not self.validate_conversation_flow(conv_idx, messages):
            valid = False
        
        # 检查 tools 字段（可选）
        if "tools" in conversation:
            tools = conversation["tools"]
            if not isinstance(tools, list):
                self.log_error(conv_idx, "'tools' 必须是数组")
                valid = False
            else:
                for tool_idx, tool in enumerate(tools):
                    if not self.validate_tool_definition(conv_idx, tool_idx, tool):
                        valid = False
        
        return valid
    
    def validate_conversation_flow(self, conv_idx: int, messages: List[Dict[str, Any]]) -> bool:
        """验证对话流程的合理性"""
        valid = True
        
        # 检查第一条消息（通常应该是 system 或 user）
        first_role = messages[0].get("role")
        if first_role not in {"system", "user"}:
            self.log_warning(conv_idx, f"第一条消息的角色是 '{first_role}'，通常应该是 'system' 或 'user'")
        
        # 检查最后一条消息（应该是 assistant）
        last_role = messages[-1].get("role")
        if last_role != "assistant":
            self.log_warning(conv_idx, f"最后一条消息的角色是 '{last_role}'，应该是 'assistant'")
        
        # 检查 tool_call 和 tool response 的配对
        tool_call_ids = set()
        tool_response_ids = set()
        
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg.get("tool_calls", []):
                    tool_call_ids.add(tc.get("id"))
            
            if msg.get("role") == "tool":
                tool_response_ids.add(msg.get("tool_call_id"))
        
        # 检查是否有未响应的 tool_call
        unresponded_calls = tool_call_ids - tool_response_ids
        if unresponded_calls:
            self.log_warning(conv_idx, f"存在未响应的 tool_call: {unresponded_calls}")
        
        # 检查是否有无对应 tool_call 的 tool response
        orphan_responses = tool_response_ids - tool_call_ids
        if orphan_responses:
            self.log_warning(conv_idx, f"存在无对应 tool_call 的 tool response: {orphan_responses}")
        
        return valid
    
    def validate_tool_definition(self, conv_idx: int, tool_idx: int, tool: Dict[str, Any]) -> bool:
        """验证工具定义格式"""
        valid = True
        
        if not isinstance(tool, dict):
            self.log_error(conv_idx, f"工具 {tool_idx}: 工具定义必须是对象")
            return False
        
        # 检查 type 字段
        if tool.get("type") != "function":
            self.log_error(conv_idx, f"工具 {tool_idx}: type 必须是 'function'")
            valid = False
        
        # 检查 function 字段
        if "function" not in tool:
            self.log_error(conv_idx, f"工具 {tool_idx}: 缺少 'function' 字段")
            valid = False
        else:
            function = tool["function"]
            if not isinstance(function, dict):
                self.log_error(conv_idx, f"工具 {tool_idx}: function 必须是对象")
                valid = False
            else:
                # 检查 function 的必需字段
                if "name" not in function:
                    self.log_error(conv_idx, f"工具 {tool_idx}: function 缺少 'name' 字段")
                    valid = False
        
        return valid
    
    def validate_file(self, file_path: Path) -> bool:
        """验证整个文件"""
        print(f"\n{'='*60}")
        print(f"验证文件: {file_path}")
        print(f"{'='*60}\n")
        
        # 读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return False
        
        # 检查顶层格式
        if not isinstance(data, list):
            print("❌ 数据格式错误: 顶层必须是数组")
            return False
        
        self.stats["total_conversations"] = len(data)
        
        if len(data) == 0:
            print("⚠️  警告: 文件中没有对话数据")
            return False
        
        print(f"📊 总对话数: {len(data)}\n")
        
        # 验证每个对话
        all_valid = True
        for conv_idx, conversation in enumerate(data):
            if not isinstance(conversation, dict):
                self.log_error(conv_idx, "对话必须是对象")
                all_valid = False
                continue
            
            if not self.validate_conversation(conv_idx, conversation):
                all_valid = False
        
        # 计算统计数据
        if self.stats["total_conversations"] > 0:
            self.stats["avg_messages_per_conversation"] = (
                self.stats["total_messages"] / self.stats["total_conversations"]
            )
        
        # 输出结果
        self.print_results(all_valid)
        
        return all_valid
    
    def print_results(self, all_valid: bool):
        """输出验证结果"""
        print(f"\n{'='*60}")
        print("验证结果")
        print(f"{'='*60}\n")
        
        # 统计信息
        print("📊 统计信息:")
        print(f"  - 总对话数: {self.stats['total_conversations']}")
        print(f"  - 总消息数: {self.stats['total_messages']}")
        print(f"  - 平均每对话消息数: {self.stats['avg_messages_per_conversation']:.2f}")
        print(f"  - Tool Calls 总数: {self.stats['tool_calls_count']}")
        print(f"  - Tool Responses 总数: {self.stats['tool_responses_count']}")
        
        print(f"\n📋 角色分布:")
        for role, count in sorted(self.stats["role_distribution"].items()):
            percentage = (count / self.stats["total_messages"]) * 100
            print(f"  - {role}: {count} ({percentage:.1f}%)")
        
        # 错误和警告
        print(f"\n🔍 验证详情:")
        print(f"  - 错误数: {len(self.errors)}")
        print(f"  - 警告数: {len(self.warnings)}")
        
        if self.errors:
            print(f"\n❌ 错误列表 (显示前10条):")
            for error in self.errors[:10]:
                print(f"  {error}")
            if len(self.errors) > 10:
                print(f"  ... 还有 {len(self.errors) - 10} 条错误")
        
        if self.warnings:
            print(f"\n⚠️  警告列表 (显示前10条):")
            for warning in self.warnings[:10]:
                print(f"  {warning}")
            if len(self.warnings) > 10:
                print(f"  ... 还有 {len(self.warnings) - 10} 条警告")
        
        # 最终结果
        print(f"\n{'='*60}")
        if all_valid and len(self.errors) == 0:
            print("✅ 验证通过！所有对话格式符合 OpenAI 要求")
        elif len(self.errors) == 0:
            print("⚠️  验证通过，但存在一些警告")
        else:
            print("❌ 验证失败，存在格式错误")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="验证 OpenAI 格式的训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -i data.json
  %(prog)s --input data.json --verbose
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="输入的 JSON 文件路径"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细的验证过程"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not args.input.exists():
        print(f"❌ 错误: 文件不存在: {args.input}")
        sys.exit(1)
    
    # 创建验证器并验证
    validator = OpenAIFormatValidator(verbose=args.verbose)
    success = validator.validate_file(args.input)
    
    # 返回适当的退出码
    sys.exit(0 if success and len(validator.errors) == 0 else 1)


if __name__ == "__main__":
    main()
