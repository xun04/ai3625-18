# 数据处理脚本说明

## 概述

本目录包含用于处理 SII 对话数据的完整管道脚本。

## 目录结构

```
data/
├── raw/<日期>/              # 原始 trajectory JSON 文件
├── first_process/<日期>/    # 第一次处理后的数据
├── second_process/<日期>/   # 第二次处理后的数据
│   └── training/            # 合并后的所有批次数据
└── parquet/<日期>/          # 各批次的 Parquet 文件
    └── training/            # 合并后的 Parquet 文件
```

## 主要脚本

### 1. `data_all.sh` - 完整数据处理管道

**功能**: 处理单个批次的数据，并自动合并所有批次

**用法**:
```bash
./scripts/data_all.sh <日期>
```

**示例**:
```bash
./scripts/data_all.sh 1104
```

**处理流程**:
1. **步骤 1**: 转换原始数据为 OpenAI 格式 (`sii_to_openai.py`)
2. **步骤 2**: 第一次处理 - 修剪尾部消息 (`first_proces_openai.py`)
3. **步骤 3**: 第二次处理 - 过滤训练数据 (`second_process_openai.py`)
4. **步骤 4**: 转换为 Parquet 格式 (`messages_json2parquet_128k.py`)
5. **步骤 5**: 合并所有批次的 JSON 数据 (`merge_all_batches.py`)
6. **步骤 6**: 将合并后的 JSON 转换为 Parquet (`messages_json2parquet_128k.py`)

**输出文件**:
- 批次文件: `data/parquet/<日期>/train_<日期>_128k.parquet`
- 合并 JSON: `data/second_process/training/training_all.json`
- 合并 Parquet: `data/parquet/training/training_all.parquet`

### 2. `merge_all_batches.py` - 合并所有批次数据

**功能**: 扫描所有日期目录，合并第二次处理后的 JSON 数据

**用法**:
```bash
python3 scripts/merge_all_batches.py
```

**说明**: 通常不需要单独运行，`data_all.sh` 会自动调用

### 3. 其他脚本

- `sii_to_openai.py`: 将 SII trajectory 格式转换为 OpenAI 格式
- `first_proces_openai.py`: 修剪对话尾部，确保以 assistant 消息结束
- `second_process_openai.py`: 过滤短对话（默认最小 5 条消息）
- `messages_json2parquet_128k.py`: 将 JSON 转换为 Parquet 格式

## 工作流程示例

### 处理新批次数据

1. 将原始数据放入 `data/raw/1105/`
2. 运行处理脚本:
   ```bash
   ./scripts/data_all.sh 1105
   ```
3. 脚本会自动:
   - 处理 1105 批次
   - 合并所有批次（包括之前的 1104 等）
   - 生成最新的训练数据

### 查看合并结果

合并后的训练数据位于:
- JSON: `data/second_process/training/training_all.json`
- Parquet: `data/parquet/training/training_all.parquet`

## 配置参数

可在 `data_all.sh` 中修改以下参数:

```bash
TOKENIZER_PATH="/path/to/tokenizer"  # Tokenizer 路径
MIN_MESSAGES=5                        # 最小消息数阈值
MAX_TOKENS=120000                     # 最大 token 数
DUPLICATE_TIMES=1                     # 数据重复次数
```

## 注意事项

1. 确保原始数据目录存在且包含 JSON 文件
2. 需要安装必要的 Python 依赖（transformers, pyarrow 等）
3. 每次运行 `data_all.sh` 都会更新合并后的训练数据
4. 各批次的数据会保留，方便对比不同批次的效果
