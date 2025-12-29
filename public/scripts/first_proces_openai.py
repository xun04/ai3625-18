"""Ensure conversations terminate with an assistant message without tool calls.

Usage
-----
python ensure_assistant_tail.py \
    --input limi-1023/training.json \
    --output limi-1023/training_assistant_tail.json

If ``--output`` is omitted, results are written to ``<input>_assistant_tail.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune trailing messages so each conversation ends with an assistant message without tool calls")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("limi-1023/training.json"),
        help="Input JSON path (relative to sft/dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to <input>_assistant_tail.json",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for the output JSON (default: 2)",
    )
    return parser.parse_args()


def prune_to_assistant_without_tools(messages: List[Any]) -> Tuple[int, bool]:
    if not isinstance(messages, list):
        return 0, False

    removed = 0
    while messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "assistant" and not last.get("tool_calls"):
            return removed, True
        messages.pop()
        removed += 1
    return removed, False


def main() -> None:
    args = parse_args()

    input_path = (Path(__file__).parent / args.input).resolve()
    if args.output:
        output_path = (Path(__file__).parent / args.output).resolve()
    else:
        default_name = input_path.stem + "_assistant_tail.json"
        output_path = input_path.with_name(default_name)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {input_path}")

    trimmed_messages = 0
    dropped_conversations = 0
    cleaned_data: List[Dict[str, Any]] = []

    for conversation in data:
        if not isinstance(conversation, dict):
            cleaned_data.append(conversation)
            continue

        messages: Any = conversation.get("messages")
        if not isinstance(messages, list):
            cleaned_data.append(conversation)
            continue

        removed, valid = prune_to_assistant_without_tools(messages)
        trimmed_messages += removed
        if valid:
            cleaned_data.append(conversation)
        else:
            dropped_conversations += 1

    output_path.write_text(
        json.dumps(cleaned_data, ensure_ascii=False, indent=args.indent),
        encoding="utf-8",
    )

    print(f"Written cleaned data to {output_path}")
    print(f"Trimmed trailing messages: {trimmed_messages}")
    if dropped_conversations:
        print(f"Dropped conversations without assistant messages: {dropped_conversations}")


if __name__ == "__main__":
    main()
