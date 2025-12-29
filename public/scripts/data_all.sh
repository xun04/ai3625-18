#!/bin/bash

# =============================================================================
# 完整数据处理管道脚本
# 流程: Raw Data -> OpenAI Format -> First Process -> Second Process -> Parquet
# 用法: ./data_all.sh <日期>
# 示例: ./data_all.sh 20251104
# =============================================================================

set -e  # 遇到错误立即退出

# 检查日期参数
if [ -z "$1" ]; then
    echo "错误: 缺少日期参数"
    echo "用法: $0 <日期>"
    echo "示例: $0 20251104"
    exit 1
fi

BATCH_DATE="$1"

# 不需要base
# 初始化 conda 以便在脚本中使用
# eval "$(conda shell.bash hook)"
source /root/anaconda3/bin/deactivate

# 配置路径 - 请根据需要修改这些路径
PROJECT_ROOT="/inspire/hdd/project/qproject-fundationmodel/public/xy/agentic-flywheel"
RAW_DIR="$PROJECT_ROOT/data/raw/$BATCH_DATE"
FIRST_PROCESS_DIR="$PROJECT_ROOT/data/first_process/$BATCH_DATE"
SECOND_PROCESS_DIR="$PROJECT_ROOT/data/second_process/$BATCH_DATE"
PARQUET_DIR="$PROJECT_ROOT/data/parquet/$BATCH_DATE"

# 合并后的训练数据路径
TRAINING_DIR="$PROJECT_ROOT/data/second_process/training"
TRAINING_PARQUET_DIR="$PROJECT_ROOT/data/parquet/training"

# Tokenizer 路径 - 用于最后的 Parquet 转换
TOKENIZER_PATH="/inspire/hdd/project/qproject-fundationmodel/public/cache/GLM-4.6"

# 脚本路径
SCRIPT_DIR="$PROJECT_ROOT/scripts"

# 中间文件和输出文件路径
OPENAI_JSON="$FIRST_PROCESS_DIR/openai_training_data.json"
FIRST_PROCESSED_JSON="$FIRST_PROCESS_DIR/openai_training_data_assistant_tail.json"
SECOND_PROCESSED_JSON="$SECOND_PROCESS_DIR/training_filtered.json"
OUTPUT_PARQUET="$PARQUET_DIR/train_${BATCH_DATE}_128k.parquet"

# 合并后的文件路径
MERGED_JSON="$TRAINING_DIR/training_all.json"
MERGED_PARQUET="$TRAINING_PARQUET_DIR/training_all.parquet"

# 处理参数
MIN_MESSAGES=5           # 第二次处理的最小消息数阈值
MAX_TOKENS=120000        # Parquet 转换的最大 token 数
DUPLICATE_TIMES=1        # 数据重复次数

echo "========================================="
echo "开始执行完整数据处理管道"
echo "========================================="
echo "批次日期: $BATCH_DATE"
echo "项目根目录: $PROJECT_ROOT"
echo "原始数据目录: $RAW_DIR"
echo "第一次处理目录: $FIRST_PROCESS_DIR"
echo "第二次处理目录: $SECOND_PROCESS_DIR"
echo "Parquet输出目录: $PARQUET_DIR"
echo "Tokenizer路径: $TOKENIZER_PATH"
echo

# 检查原始数据目录
if [ ! -d "$RAW_DIR" ]; then
    echo "错误: 原始数据目录不存在: $RAW_DIR"
    exit 1
fi

# 检查原始数据目录是否有文件
if [ -z "$(ls -A $RAW_DIR/*.json 2>/dev/null)" ]; then
    echo "警告: 原始数据目录中没有找到 JSON 文件"
    echo "请确保 $RAW_DIR 中有 *.json 文件"
    exit 1
fi

# 创建必要的目录
mkdir -p "$FIRST_PROCESS_DIR"
mkdir -p "$SECOND_PROCESS_DIR"
mkdir -p "$PARQUET_DIR"
mkdir -p "$TRAINING_DIR"
mkdir -p "$TRAINING_PARQUET_DIR"

# =============================================================================
# 步骤 1: 转换原始数据为 OpenAI 格式
# =============================================================================
echo "========================================="
echo "步骤 1/4: 转换原始数据为 OpenAI 格式"
echo "========================================="
echo "输入: $RAW_DIR"
echo "输出: $OPENAI_JSON"
echo

python3 "$SCRIPT_DIR/sii_to_openai.py" \
    --input "$RAW_DIR" \
    --output "$OPENAI_JSON" \
    --indent 2

if [ $? -ne 0 ]; then
    echo "错误: sii_to_openai.py 执行失败"
    exit 1
fi

if [ ! -f "$OPENAI_JSON" ]; then
    echo "错误: OpenAI JSON 文件未生成"
    exit 1
fi

file_size=$(du -h "$OPENAI_JSON" | cut -f1)
echo "✓ 步骤 1 完成，生成文件: $OPENAI_JSON (大小: $file_size)"
echo

# =============================================================================
# 步骤 2: 第一次处理 - 修剪尾部消息
# =============================================================================
echo "========================================="
echo "步骤 2/4: 第一次处理 - 修剪尾部消息"
echo "========================================="
echo "输入: $OPENAI_JSON"
echo "输出: $FIRST_PROCESSED_JSON"
echo

python3 "$SCRIPT_DIR/first_proces_openai.py" \
    --input "$OPENAI_JSON" \
    --output "$FIRST_PROCESSED_JSON" \
    --indent 2

if [ $? -ne 0 ]; then
    echo "错误: first_proces_openai.py 执行失败"
    exit 1
fi

if [ ! -f "$FIRST_PROCESSED_JSON" ]; then
    echo "错误: 第一次处理后的 JSON 文件未生成"
    exit 1
fi

file_size=$(du -h "$FIRST_PROCESSED_JSON" | cut -f1)
echo "✓ 步骤 2 完成，生成文件: $FIRST_PROCESSED_JSON (大小: $file_size)"
echo

# =============================================================================
# 步骤 3: 第二次处理 - 过滤训练数据
# =============================================================================
echo "========================================="
echo "步骤 3/4: 第二次处理 - 过滤训练数据"
echo "========================================="
echo "输入: $FIRST_PROCESSED_JSON"
echo "输出: $SECOND_PROCESSED_JSON"
echo "最小消息数阈值: $MIN_MESSAGES"
echo

python3 "$SCRIPT_DIR/second_process_openai.py" \
    --input "$FIRST_PROCESSED_JSON" \
    --output "$SECOND_PROCESSED_JSON" \
    --min_messages "$MIN_MESSAGES"

if [ $? -ne 0 ]; then
    echo "错误: second_process_openai.py 执行失败"
    exit 1
fi

if [ ! -f "$SECOND_PROCESSED_JSON" ]; then
    echo "错误: 第二次处理后的 JSON 文件未生成"
    exit 1
fi

file_size=$(du -h "$SECOND_PROCESSED_JSON" | cut -f1)
echo "✓ 步骤 3 完成，生成文件: $SECOND_PROCESSED_JSON (大小: $file_size)"
echo

# =============================================================================
# 步骤 4: 转换为 Parquet 格式
# =============================================================================
echo "========================================="
echo "步骤 4/4: 转换为 Parquet 格式"
echo "========================================="
echo "输入: $SECOND_PROCESSED_JSON"
echo "输出: $OUTPUT_PARQUET"
echo "最大 Token 数: $MAX_TOKENS"
echo "重复次数: $DUPLICATE_TIMES"
echo

python3 "$SCRIPT_DIR/messages_json2parquet_128k.py" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --input_json "$SECOND_PROCESSED_JSON" \
    --output_parquet "$OUTPUT_PARQUET" \
    --max_tokens "$MAX_TOKENS" \
    --duplicate_times "$DUPLICATE_TIMES"

if [ $? -ne 0 ]; then
    echo "错误: messages_json2parquet_128k.py 执行失败"
    exit 1
fi

if [ ! -f "$OUTPUT_PARQUET" ]; then
    echo "错误: Parquet 文件未生成"
    exit 1
fi

file_size=$(du -h "$OUTPUT_PARQUET" | cut -f1)
echo "✓ 步骤 4 完成，生成文件: $OUTPUT_PARQUET (大小: $file_size)"
echo

# =============================================================================
# 步骤 5: 合并所有批次的 JSON 数据
# =============================================================================
echo "========================================="
echo "步骤 5/6: 合并所有批次的 JSON 数据"
echo "========================================="
echo "扫描目录: $PROJECT_ROOT/data/second_process/"
echo "输出文件: $MERGED_JSON"
echo

python3 "$SCRIPT_DIR/merge_all_batches.py"

if [ $? -ne 0 ]; then
    echo "错误: merge_all_batches.py 执行失败"
    exit 1
fi

if [ ! -f "$MERGED_JSON" ]; then
    echo "错误: 合并后的 JSON 文件未生成"
    exit 1
fi

merged_json_size=$(du -h "$MERGED_JSON" | cut -f1)
echo "✓ 步骤 5 完成，生成文件: $MERGED_JSON (大小: $merged_json_size)"
echo

# =============================================================================
# 步骤 6: 将合并后的 JSON 转换为 Parquet
# =============================================================================
echo "========================================="
echo "步骤 6/6: 将合并后的 JSON 转换为 Parquet"
echo "========================================="
echo "输入: $MERGED_JSON"
echo "输出: $MERGED_PARQUET"
echo "最大 Token 数: $MAX_TOKENS"
echo "重复次数: $DUPLICATE_TIMES"
echo

python3 "$SCRIPT_DIR/messages_json2parquet_128k.py" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --input_json "$MERGED_JSON" \
    --output_parquet "$MERGED_PARQUET" \
    --max_tokens "$MAX_TOKENS" \
    --duplicate_times "$DUPLICATE_TIMES"

if [ $? -ne 0 ]; then
    echo "错误: 合并后的 Parquet 转换失败"
    exit 1
fi

if [ ! -f "$MERGED_PARQUET" ]; then
    echo "错误: 合并后的 Parquet 文件未生成"
    exit 1
fi

merged_parquet_size=$(du -h "$MERGED_PARQUET" | cut -f1)
echo "✓ 步骤 6 完成，生成文件: $MERGED_PARQUET (大小: $merged_parquet_size)"
echo

# =============================================================================
# 完成总结
# =============================================================================
echo "========================================="
echo "数据处理管道执行完成！"
echo "========================================="
echo "批次 $BATCH_DATE 的文件:"
echo "  1. OpenAI 格式:        $OPENAI_JSON"
echo "  2. 第一次处理后:       $FIRST_PROCESSED_JSON"
echo "  3. 第二次处理后:       $SECOND_PROCESSED_JSON"
echo "  4. Parquet 文件:       $OUTPUT_PARQUET (大小: $file_size)"
echo
echo "合并后的训练数据:"
echo "  5. 合并 JSON:          $MERGED_JSON (大小: $merged_json_size)"
echo "  6. 合并 Parquet:       $MERGED_PARQUET (大小: $merged_parquet_size)"
echo
echo "所有步骤已成功完成！"
echo "========================================="
