"""Convert LIMI trajectory logs into OpenAI chat completion training format.

Usage
-----
python convert_to_openai.py --input /path/to/trajectory.json --output output.json

When ``--input`` points to a directory, all ``*.json`` files inside (non-recursively)
are converted and aggregated. If ``--output`` refers to a directory, a file with the
same stem as each input (plus ``_openai.json``) is written in that directory. If it
refers to a file, all converted conversations are written into that single JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input trajectory file or directory")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file or directory for converted data",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON dumps (default: 2)",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="When set, write one OpenAI JSON per input file into the output directory.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="openai_training_data.json",
        help="Filename to use when writing aggregated output into a directory (default: openai_training_data.json)",
    )
    return parser.parse_args()


def normalize_schema(obj: Any) -> Any:
    """Recursively normalise JSON schema type casing to match OpenAI expectations."""

    if isinstance(obj, dict):
        normalized: Dict[str, Any] = {}
        for key, value in obj.items():
            if key == "type" and isinstance(value, str):
                normalized[key] = value.lower()
            else:
                normalized[key] = normalize_schema(value)
        return normalized
    if isinstance(obj, list):
        return [normalize_schema(item) for item in obj]
    return obj


def convert_tool_definitions(raw_tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if not raw_tools:
        return tools

    for tool in raw_tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        parameters = tool.get("parameters") or {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": normalize_schema(parameters),
                },
            }
        )
    return tools


def ensure_arguments_string(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return json.dumps({}, ensure_ascii=False)
    return json.dumps(arguments, ensure_ascii=False)


def convert_events_to_conversation(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    system_instruction = None
    available_tools: Optional[List[Dict[str, Any]]] = None
    messages: List[Dict[str, Any]] = []

    i = 0
    n = len(events)
    while i < n:
        event = events[i]
        event_type = event.get("event_type")

        if event_type == "system_event":
            metadata = event.get("metadata", {})
            system_instruction = metadata.get("system_instruction") or system_instruction
            available_tools = metadata.get("available_tools", available_tools)
            i += 1
            continue

        if event_type == "user_query":
            messages.append({"role": "user", "content": event.get("query", "")})
            i += 1
            continue

        if event_type == "assistant_response":
            response = event.get("response")
            tool_calls_from_event = event.get("tool_calls") or []
            if tool_calls_from_event:
                tool_calls = [
                    {
                        "id": call.get("id") or call.get("tool_call_id") or call.get("toolCallId") or "call",
                        "type": "function",
                        "function": {
                            "name": call.get("tool_name") or call.get("name") or "",
                            "arguments": ensure_arguments_string(call.get("arguments")),
                        },
                    }
                    for call in tool_calls_from_event
                ]
                messages.append({"role": "assistant", "content": response or "", "tool_calls": tool_calls})
                i += 1
                continue

            # Look ahead for dedicated tool_call events when response is empty.
            if not response:
                j = i + 1
                tool_calls: List[Dict[str, Any]] = []
                while j < n and events[j].get("event_type") == "tool_call":
                    call_event = events[j]
                    tool_calls.append(
                        {
                            "id": call_event.get("tool_call_id") or call_event.get("id") or f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {
                                "name": call_event.get("tool_name") or "",
                                "arguments": ensure_arguments_string(call_event.get("tool_args")),
                            },
                        }
                    )
                    j += 1
                if tool_calls:
                    messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                    i = j
                    continue

            messages.append({"role": "assistant", "content": response or ""})
            i += 1
            continue

        if event_type == "tool_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": event.get("tool_call_id") or event.get("id") or "call",
                            "type": "function",
                            "function": {
                                "name": event.get("tool_name") or "",
                                "arguments": ensure_arguments_string(event.get("tool_args")),
                            },
                        }
                    ],
                }
            )
            i += 1
            continue

        if event_type == "tool_result":
            content = event.get("tool_result")
            if content is None:
                content = event.get("result") or ""
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.get("tool_call_id") or event.get("toolCallId"),
                    "content": content,
                }
            )
            i += 1
            continue

        # Skip any other event types.
        i += 1

    if system_instruction:
        messages.insert(0, {"role": "system", "content": system_instruction})

    tools = convert_tool_definitions(available_tools)
    return {"tools": tools, "messages": messages}


def load_events(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def convert_file(path: Path) -> List[Dict[str, Any]]:
    events = load_events(path)
    
    if isinstance(events, dict) and events and "data" in events:
        events=events["data"]
    
    if isinstance(events, list) and events and isinstance(events[0], dict) and "messages" in events[0]:
        # Already in OpenAI training format
        return events  # type: ignore[return-value]
    
    if not isinstance(events, list):
        raise ValueError(f"Expected list of events in {path}")

    return [convert_events_to_conversation(events)]


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for file_path in sorted(input_path.glob("*.json")):
        if file_path.is_file():
            yield file_path


def write_output(data: List[Dict[str, Any]], output_path: Path, indent: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def main() -> None:
    args = parse_args()
    input_files = list(iter_input_files(args.input))
    if not input_files:
        raise FileNotFoundError(f"No input JSON files found under {args.input}")

    if args.per_file:
        output_dir = args.output
        if not output_dir.suffix:
            output_dir.mkdir(parents=True, exist_ok=True)
        elif output_dir.is_dir():
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("--per-file requires --output to be a directory")

        # regroup per input file again for writing individually
        for file_path in input_files:
            conversations = convert_file(file_path)
            out_path = output_dir / f"{file_path.stem}_openai.json"
            write_output(conversations, out_path, args.indent)
        return

    aggregated: List[Dict[str, Any]] = []
    for file_path in input_files:
        aggregated.extend(convert_file(file_path))

    if args.output.suffix:
        write_output(aggregated, args.output, args.indent)
        return

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_filename
    write_output(aggregated, out_path, args.indent)


if __name__ == "__main__":
    main()
