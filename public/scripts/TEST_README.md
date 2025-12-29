# OpenAI 格式数据验证测试

## 概述

`test_openai_format.py` 是一个用于验证训练数据是否符合 OpenAI 对话格式要求的单元测试脚本。

## 功能特性

### 验证项目

1. **数据结构验证**
   - 顶层必须是数组
   - 每个对话必须是对象
   - 每个对话必须包含 `messages` 字段

2. **消息格式验证**
   - `role` 字段必须是 `system`、`user`、`assistant` 或 `tool` 之一
   - 非 `tool` 角色必须有 `content` 字段
   - `tool` 角色必须有 `tool_call_id` 和 `content` 字段

3. **Tool Call 验证**
   - 只有 `assistant` 消息可以包含 `tool_calls`
   - `tool_calls` 必须是数组
   - 每个 tool_call 必须包含 `id`、`type` 和 `function` 字段
   - `type` 必须是 `"function"`
   - `function` 必须包含 `name` 和 `arguments` 字段
   - `arguments` 必须是字符串或字典

4. **对话流程验证**
   - 第一条消息通常应该是 `system` 或 `user`
   - 最后一条消息应该是 `assistant`
   - 检查 tool_call 和 tool response 的配对关系

5. **工具定义验证**（如果存在 `tools` 字段）
   - 工具定义必须包含 `type` 和 `function` 字段
   - `type` 必须是 `"function"`
   - `function` 必须包含 `name` 字段

### 统计信息

脚本会输出以下统计信息：
- 总对话数
- 总消息数
- 平均每对话消息数
- Tool Calls 总数
- Tool Responses 总数
- 角色分布（每种角色的消息数和百分比）

## 使用方法

### 基本用法

```bash
python scripts/test_openai_format.py --input <path_to_json_file>
```

### 示例

```bash
# 验证处理后的数据
python scripts/test_openai_format.py --input data/second_process/20251223/training_filtered.json

# 显示详细的验证过程
python scripts/test_openai_format.py --input data/second_process/20251223/training_filtered.json --verbose

# 使用虚拟环境
source /data2/ai3625/public/dataclean/bin/activate
python scripts/test_openai_format.py -i data/second_process/20251223/training_filtered.json
```

### 命令行参数

- `--input`, `-i`: (必需) 输入的 JSON 文件路径
- `--verbose`, `-v`: (可选) 显示详细的验证过程，包括每个错误和警告的实时输出

### 退出码

- `0`: 验证通过，无错误
- `1`: 验证失败，存在格式错误

## 输出说明

### 成功示例

```
============================================================
验证文件: data/second_process/20251223/training_filtered.json
============================================================

📊 总对话数: 3

============================================================
验证结果
============================================================

📊 统计信息:
  - 总对话数: 3
  - 总消息数: 114
  - 平均每对话消息数: 38.00
  - Tool Calls 总数: 34
  - Tool Responses 总数: 34

📋 角色分布:
  - assistant: 68 (59.6%)
  - system: 3 (2.6%)
  - tool: 34 (29.8%)
  - user: 9 (7.9%)

🔍 验证详情:
  - 错误数: 0
  - 警告数: 0

============================================================
✅ 验证通过！所有对话格式符合 OpenAI 要求
============================================================
```

### 错误示例

当存在格式错误时，会显示详细的错误信息：

```
❌ 错误列表 (显示前10条):
  [对话 0] 消息 5: tool_call 缺少字段: {'id'}
  [对话 1] 消息 2: 非法的 role 值 'invalid', 必须是 {'system', 'user', 'assistant', 'tool'} 之一
  [对话 2] 消息 8: tool 消息缺少字段: {'tool_call_id'}
  ...
```

### 警告示例

警告不会导致验证失败，但会提示可能的问题：

```
⚠️  警告列表 (显示前10条):
  [对话 0] 第一条消息的角色是 'assistant'，通常应该是 'system' 或 'user'
  [对话 1] 最后一条消息的角色是 'user'，应该是 'assistant'
  [对话 2] 存在未响应的 tool_call: {'call_123'}
  ...
```

## 集成到工作流

### 在数据处理管道中使用

可以在 `data_all.sh` 脚本中添加验证步骤：

```bash
# 步骤 3 之后，添加验证
echo "验证数据格式..."
python3 "$SCRIPT_DIR/test_openai_format.py" \
    --input "$SECOND_PROCESSED_JSON"

if [ $? -ne 0 ]; then
    echo "警告: 数据格式验证失败，但继续处理"
fi
```

### 作为 Git 钩子

可以将验证脚本添加到 Git pre-commit 钩子中：

```bash
#!/bin/bash
# .git/hooks/pre-commit

python scripts/test_openai_format.py --input data/second_process/training/training_all.json
if [ $? -ne 0 ]; then
    echo "数据格式验证失败，提交被阻止"
    exit 1
fi
```

### 在 CI/CD 中使用

```yaml
# .github/workflows/validate-data.yml
name: Validate Training Data

on:
  push:
    paths:
      - 'data/second_process/**/*.json'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Validate OpenAI Format
        run: |
          python scripts/test_openai_format.py \
            --input data/second_process/training/training_all.json
```

## 常见问题

### Q: 为什么会出现 "存在未响应的 tool_call" 警告？

A: 这表示某个 assistant 消息中调用了工具，但没有对应的 tool 消息返回结果。这通常不是错误，但可能表示对话被截断了。

### Q: 为什么最后一条消息不是 assistant？

A: OpenAI 的训练格式要求对话以 assistant 的回复结束。如果最后一条消息是 user 或 tool，数据处理管道应该已经修剪了这些消息（通过 `first_proces_openai.py`）。

### Q: arguments 应该是字符串还是字典？

A: 两种格式都支持。如果是字符串，应该是有效的 JSON 字符串；如果是字典，会在使用时自动转换为 JSON 字符串。

## 扩展和自定义

如果需要添加自定义验证规则，可以修改 `OpenAIFormatValidator` 类：

```python
def validate_custom_rule(self, conv_idx: int, conversation: Dict[str, Any]) -> bool:
    """自定义验证规则"""
    # 实现你的验证逻辑
    pass

# 在 validate_conversation 方法中调用
if not self.validate_custom_rule(conv_idx, conversation):
    valid = False
```

## 相关脚本

- `sii_to_openai.py`: 转换原始数据为 OpenAI 格式
- `first_proces_openai.py`: 修剪对话尾部
- `second_process_openai.py`: 过滤和合并消息
- `messages_json2parquet_128k.py`: 转换为 Parquet 格式

## 维护者

如有问题或建议，请联系项目维护者。
