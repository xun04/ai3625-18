"""Synthesize OpenAI chat training data from a benchmark dataset.

This script reads benchmark records (JSON list or JSONL) and emits a JSONL
file where each line is a training sample in OpenAI chat format.

Example:
    python public/scripts/synthesize_benchmark_training_data.py \
        --input public/swebench-verified/test/data-00000-of-00001.json \
        --output public/data/second_process/openai_training_data.jsonl \
        --min-samples 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Benchmark JSON/JSONL file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum number of samples to emit.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on emitted samples.")
    parser.add_argument(
        "--variants-per-record",
        type=int,
        default=3,
        help="How many prompt variants to emit per benchmark record.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for shuffling and sampling.",
    )
    parser.add_argument(
        "--min-problem-length",
        type=int,
        default=80,
        help="Minimum number of characters required in the problem statement.",
    )
    parser.add_argument(
        "--max-patch-lines",
        type=int,
        default=800,
        help="Maximum number of lines allowed in the patch.",
    )
    parser.add_argument(
        "--answer-field",
        type=str,
        default="patch",
        help="Field name to use as assistant answer (default: patch).",
    )
    parser.add_argument(
        "--require-diff",
        action="store_true",
        help="Only keep samples whose answer contains a unified diff header.",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Balance samples across difficulty and patch-size buckets.",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=3.0,
        help="Minimum quality score required to keep a sample.",
    )
    parser.add_argument(
        "--max-per-repo",
        type=int,
        default=10,
        help="Maximum samples to keep per repository to improve diversity.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=(
            "You are a senior software engineer. Given a benchmark issue description and "
            "context, produce a patch that resolves the problem."
        ),
        help="System prompt to inject at the start of each conversation.",
    )
    return parser.parse_args()


def load_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError(f"Unsupported JSON structure in {path}")


def load_jsonl_file(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_input_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for file_path in sorted(input_path.glob("*.json")):
        if file_path.is_file():
            yield file_path
    for file_path in sorted(input_path.glob("*.jsonl")):
        if file_path.is_file():
            yield file_path


def load_records(input_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for file_path in iter_input_files(input_path):
        if file_path.suffix == ".jsonl":
            records.extend(load_jsonl_file(file_path))
        else:
            records.extend(load_json_file(file_path))
    if not records:
        raise FileNotFoundError(f"No benchmark records found under {input_path}")
    return records


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_sections(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        "repo": normalize_text(record.get("repo")),
        "instance_id": normalize_text(record.get("instance_id")),
        "problem_statement": normalize_text(record.get("problem_statement")),
        "hints_text": normalize_text(record.get("hints_text")),
        "fail_to_pass": normalize_text(record.get("FAIL_TO_PASS")),
        "pass_to_pass": normalize_text(record.get("PASS_TO_PASS")),
        "base_commit": normalize_text(record.get("base_commit")),
        "environment_setup_commit": normalize_text(record.get("environment_setup_commit")),
    }


def build_prompt_variants(sections: Dict[str, str]) -> List[Tuple[str, str]]:
    repo_line = f"Repository: {sections['repo']}" if sections["repo"] else ""
    instance_line = f"Instance ID: {sections['instance_id']}" if sections["instance_id"] else ""
    base_commit = f"Base Commit: {sections['base_commit']}" if sections["base_commit"] else ""
    env_commit = (
        f"Environment Setup Commit: {sections['environment_setup_commit']}"
        if sections["environment_setup_commit"]
        else ""
    )
    hints = f"Hints:\n{sections['hints_text']}" if sections["hints_text"] else ""
    failing = f"Failing Tests:\n{sections['fail_to_pass']}" if sections["fail_to_pass"] else ""
    passing = f"Passing Tests:\n{sections['pass_to_pass']}" if sections["pass_to_pass"] else ""
    statement = sections["problem_statement"]

    variants: List[Tuple[str, str]] = []
    if statement:
        variants.append(
            (
                "full_context",
                "\n\n".join(
                    part
                    for part in [
                        repo_line,
                        instance_line,
                        "Problem Statement:\n" + statement,
                        hints,
                        failing,
                        passing,
                        base_commit,
                        env_commit,
                    ]
                    if part
                ),
            )
        )
        variants.append(
            (
                "problem_focused",
                "\n\n".join(
                    part
                    for part in [
                        "Problem Statement:\n" + statement,
                        hints,
                        failing,
                    ]
                    if part
                ),
            )
        )
        variants.append(
            (
                "tests_focused",
                "\n\n".join(
                    part
                    for part in [
                        repo_line,
                        "Problem Statement:\n" + statement,
                        failing,
                        passing,
                    ]
                    if part
                ),
            )
        )
    else:
        variants.append(
            (
                "minimal",
                "\n\n".join(
                    part
                    for part in [
                        repo_line,
                        instance_line,
                        hints,
                        failing,
                        passing,
                    ]
                    if part
                )
                or "Please resolve the benchmark task described in the provided record.",
            )
        )
    return variants


def select_system_prompts(base_prompt: str) -> List[str]:
    return [
        base_prompt,
        base_prompt
        + " Respond with a unified diff only, no extra commentary.",
        base_prompt
        + " Ensure the patch is minimal, focused on the root cause.",
    ]


def bucketize_patch_size(patch_lines: int) -> str:
    if patch_lines <= 80:
        return "small"
    if patch_lines <= 200:
        return "medium"
    return "large"


def bucketize_difficulty(raw: str) -> str:
    text = raw.lower()
    if "15 min" in text:
        return "short"
    if "1 hour" in text or "2 hour" in text or "2 hours" in text:
        return "medium"
    if "long" in text or "day" in text or "week" in text:
        return "long"
    return "unknown"


def count_patch_lines(patch: str) -> int:
    return len([line for line in patch.splitlines() if line.strip()])


def count_patch_files(patch: str) -> int:
    return patch.count("diff --git ")


def quality_score(sections: Dict[str, str], patch: str) -> float:
    score = 0.0
    if sections["problem_statement"]:
        score += 2.0
    if sections["fail_to_pass"]:
        score += 1.0
    if sections["pass_to_pass"]:
        score += 0.5
    if sections["hints_text"]:
        score += 0.5
    if "diff --git" in patch:
        score += 2.0
    if count_patch_files(patch) >= 1:
        score += 1.0
    return score


def is_high_quality(
    sections: Dict[str, str],
    patch: str,
    min_problem_length: int,
    max_patch_lines: int,
    require_diff: bool,
) -> bool:
    if require_diff and "diff --git" not in patch:
        return False
    if len(sections["problem_statement"]) < min_problem_length:
        return False
    if count_patch_lines(patch) > max_patch_lines:
        return False
    return True


def build_record_key(record: Dict[str, Any], patch: str, user_prompt: str) -> str:
    parts = [
        normalize_text(record.get("repo")),
        normalize_text(record.get("instance_id")),
        normalize_text(record.get("problem_statement")),
        user_prompt,
        patch,
    ]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest


def build_sample(
    record: Dict[str, Any],
    answer_field: str,
    system_prompt: str,
    user_prompt: str,
) -> Optional[Dict[str, Any]]:
    answer = record.get(answer_field)
    if not isinstance(answer, str) or not answer.strip():
        return None
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


def synthesize_records(
    records: List[Dict[str, Any]],
    min_samples: int,
    max_samples: Optional[int],
    answer_field: str,
    system_prompt: str,
    variants_per_record: int,
    seed: int,
    require_diff: bool,
    min_problem_length: int,
    max_patch_lines: int,
    balanced_sampling: bool,
    min_quality_score: float,
    max_per_repo: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    synthesized: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    system_prompts = select_system_prompts(system_prompt)
    seen_keys: set[str] = set()
    repo_counts: Dict[str, int] = {}
    for record in records:
        answer = record.get(answer_field)
        if not isinstance(answer, str):
            continue
        sections = build_sections(record)
        if not is_high_quality(
            sections,
            answer,
            min_problem_length=min_problem_length,
            max_patch_lines=max_patch_lines,
            require_diff=require_diff,
        ):
            continue
        variants = build_prompt_variants(sections)
        if not variants:
            continue
        rng.shuffle(variants)
        prompt_variants = variants[: max(1, variants_per_record)]
        for idx, (_, user_prompt) in enumerate(prompt_variants):
            system_variant = system_prompts[idx % len(system_prompts)]
            sample = build_sample(record, answer_field, system_variant, user_prompt)
            if sample:
                record_key = build_record_key(record, answer, user_prompt)
                if record_key in seen_keys:
                    continue
                seen_keys.add(record_key)
                quality = quality_score(sections, answer)
                if quality < min_quality_score:
                    continue
                repo_name = sections["repo"] or "unknown"
                if max_per_repo > 0:
                    if repo_counts.get(repo_name, 0) >= max_per_repo:
                        continue
                    repo_counts[repo_name] = repo_counts.get(repo_name, 0) + 1
                meta = {
                    "repo": sections["repo"],
                    "instance_id": sections["instance_id"],
                    "difficulty": normalize_text(record.get("difficulty")),
                    "difficulty_bucket": bucketize_difficulty(
                        normalize_text(record.get("difficulty"))
                    ),
                    "patch_lines": count_patch_lines(answer),
                    "patch_bucket": bucketize_patch_size(count_patch_lines(answer)),
                    "quality_score": quality,
                }
                synthesized.append((sample, meta))

    if not synthesized:
        raise ValueError("No valid samples could be synthesized (missing answer field).")

    if balanced_sampling:
        buckets: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
        for sample, meta in synthesized:
            key = (
                str(meta.get("difficulty_bucket", "unknown")),
                str(meta.get("patch_bucket", "unknown")),
            )
            buckets.setdefault(key, []).append((sample, meta))
        for samples in buckets.values():
            rng.shuffle(samples)
        ordered: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        while buckets:
            for key in list(buckets.keys()):
                if buckets[key]:
                    ordered.append(buckets[key].pop())
                if not buckets[key]:
                    buckets.pop(key, None)
        synthesized = ordered
    else:
        rng.shuffle(synthesized)

    target = max(min_samples, 0)
    if max_samples is not None:
        target = min(target, max_samples)

    if len(synthesized) >= target:
        return [sample for sample, _ in synthesized[:target]]

    # Repeat samples with shuffle for coverage when data is sparse.
    repeated: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    while len(repeated) < target:
        rng.shuffle(synthesized)
        repeated.extend(synthesized)
    return [sample for sample, _ in repeated[:target]]


def write_jsonl(samples: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    samples = synthesize_records(
        records,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        answer_field=args.answer_field,
        system_prompt=args.system_prompt,
        variants_per_record=args.variants_per_record,
        seed=args.seed,
        require_diff=args.require_diff,
        min_problem_length=args.min_problem_length,
        max_patch_lines=args.max_patch_lines,
        balanced_sampling=args.balanced_sampling,
        min_quality_score=args.min_quality_score,
        max_per_repo=args.max_per_repo,
    )
    write_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()