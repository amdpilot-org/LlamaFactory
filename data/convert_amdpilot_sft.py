#!/usr/bin/env python3
"""Prepare amdpilot LoRA SFT data for LLaMA-Factory (v4: fixed role alternation + leak-free eval).

v4 changes from v2:
  - Removed separator/recap user messages that broke OpenAI converter validation
    (v2/v3 silently dropped 66% of training data due to this)
  - Trajectory-level train/eval split (no view of eval tasks in training)
  - Trim trailing messages to ensure even count after system extraction

Produces THREE complementary views from raw amdpilot trajectories:
  View 1 - BOOKEND: task prefix + final solution suffix (direct concat, no separator)
  View 2 - FULL: complete trajectory (truncated at training time by cutoff_len)
  View 3 - SOLUTION_CHUNKS: last N turns with task prefix (direct concat, no recap)

All use OpenAI format (role/content/tool_calls/tool_call_id).
"""

import json
import random
import sys
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "amdpilot-logs" / "lora-sft-dataset" / "combined" / "train.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent
SEED = 42
N_EVAL = 10


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def normalize_message(msg: dict) -> dict:
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", part.get("content", str(part))))
            elif isinstance(part, str):
                parts.append(part)
        content = "\n".join(str(p) for p in parts).strip()
    if not isinstance(content, str):
        content = str(content) if content else ""

    m = {"role": msg["role"], "content": content}
    if msg.get("tool_calls"):
        m["tool_calls"] = msg["tool_calls"]
    if msg.get("tool_call_id"):
        m["tool_call_id"] = msg["tool_call_id"]
    return m


def normalize_messages(messages: list[dict]) -> list[dict]:
    return [normalize_message(m) for m in messages]


def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(json.dumps(m)) for m in messages) // 4


def find_prefix_end(messages: list[dict]) -> int:
    """Find the end index of task prefix (system + first user message)."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            return i
    return min(2, len(messages))


def find_last_write_index(messages: list[dict]) -> int:
    """Find the index of the last WriteFile tool call."""
    last_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            tools = [tc["function"]["name"] for tc in m["tool_calls"] if tc and tc.get("function")]
            if "WriteFile" in tools or "Shell" in tools:
                last_idx = i
                break
    return last_idx


def ensure_valid_alternation(messages: list[dict]) -> list[dict]:
    """Trim messages to ensure valid role alternation for the OpenAI converter.

    The converter expects (after system extraction):
      positions 0, 2, 4, ...: user or tool (observation)
      positions 1, 3, 5, ...: assistant or function
    with an even total count (grouped by consecutive tool messages).

    Strategy: trim from the end until we get a valid even-count sequence
    ending with an assistant message (or its tool responses).
    """
    if not messages:
        return messages

    result = list(messages)
    for _ in range(5):
        has_system = result[0]["role"] == "system" if result else False
        after = result[1:] if has_system else result

        grouped_count = 0
        i = 0
        while i < len(after):
            if after[i]["role"] == "tool":
                while i < len(after) and after[i]["role"] == "tool":
                    i += 1
            else:
                i += 1
            grouped_count += 1

        if grouped_count >= 2 and grouped_count % 2 == 0:
            return result

        while result and result[-1]["role"] == "tool":
            result.pop()
        if result and result[-1]["role"] == "assistant":
            result.pop()

    return result if len(result) >= 3 else []


def find_assistant_at_or_before(messages: list[dict], idx: int) -> int:
    """Walk backward from idx to find an assistant message (ensures suffix starts on even slot)."""
    for i in range(idx, -1, -1):
        if messages[i]["role"] == "assistant":
            return i
    return idx


def extract_bookend(messages: list[dict]) -> list[dict]:
    """Extract PREFIX (task setup) + SUFFIX (final solution), no separator."""
    prefix_end = find_prefix_end(messages)
    prefix = messages[:prefix_end]

    last_action = find_last_write_index(messages)
    suffix_start = max(prefix_end, last_action - 2)
    suffix_start = find_assistant_at_or_before(messages, suffix_start)
    suffix = messages[suffix_start:]

    if suffix_start <= prefix_end:
        return ensure_valid_alternation(messages)

    return ensure_valid_alternation(prefix + suffix)


def extract_solution_chunk(messages: list[dict]) -> list[dict]:
    """Extract task prefix + last N turns containing the final solution, no recap."""
    prefix_end = find_prefix_end(messages)
    prefix = messages[:prefix_end]

    last_action = find_last_write_index(messages)
    chunk_start = max(prefix_end, last_action - 6)
    chunk_start = find_assistant_at_or_before(messages, chunk_start)
    chunk = messages[chunk_start:]

    if chunk_start <= prefix_end:
        return ensure_valid_alternation(messages)

    return ensure_valid_alternation(prefix + chunk)


def passes_converter_validation(messages: list[dict]) -> bool:
    """Check if messages will survive the OpenAI converter's strict role alternation."""
    roles = [m["role"] for m in messages]
    has_system = roles[0] == "system" if roles else False
    after = roles[1:] if has_system else roles
    aligned = []
    i = 0
    while i < len(after):
        if after[i] == "tool":
            while i < len(after) and after[i] == "tool":
                i += 1
            aligned.append("observation")
        else:
            aligned.append(after[i])
            i += 1
    if len(aligned) < 2 or len(aligned) % 2 != 0:
        return False
    for j, r in enumerate(aligned):
        if j % 2 == 0 and r not in ("user", "observation"):
            return False
        if j % 2 == 1 and r not in ("assistant", "function"):
            return False
    return True


def write_jsonl(data: list[dict], path: Path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Wrote {len(data)} examples to {path.name} ({size_mb:.1f} MB)")


def main():
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    examples = load_examples(DATASET_PATH)
    print(f"Loaded {len(examples)} raw trajectories from {DATASET_PATH.name}")

    # --- Trajectory-level split FIRST (before any view extraction) ---
    rng = random.Random(SEED)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    eval_indices = set(indices[:N_EVAL])
    train_indices = set(indices[N_EVAL:])

    train_examples = [examples[i] for i in sorted(train_indices)]
    eval_examples = [examples[i] for i in sorted(eval_indices)]
    print(f"Split: {len(train_examples)} train trajectories, {len(eval_examples)} eval trajectories (held out completely)")

    # --- Generate 3 views for TRAIN trajectories only ---
    train_data = []
    view_counts = {"bookend": 0, "full": 0, "solution": 0}
    dropped = {"bookend": 0, "full": 0, "solution": 0}

    for ex in train_examples:
        msgs = normalize_messages(ex["messages"])

        bookend = extract_bookend(msgs)
        if bookend:
            train_data.append({"messages": bookend})
            view_counts["bookend"] += 1
        else:
            dropped["bookend"] += 1

        full = ensure_valid_alternation(msgs)
        if full:
            train_data.append({"messages": full})
            view_counts["full"] += 1
        else:
            dropped["full"] += 1

        solution = extract_solution_chunk(msgs)
        if solution:
            train_data.append({"messages": solution})
            view_counts["solution"] += 1
        else:
            dropped["solution"] += 1

    rng.shuffle(train_data)

    pre_filter = len(train_data)
    train_data = [ex for ex in train_data if passes_converter_validation(ex["messages"])]

    print(f"\n=== Train views from {len(train_examples)} trajectories ===")
    for view, count in view_counts.items():
        d = dropped[view]
        print(f"  {view:12s}: {count} valid, {d} dropped (odd msg count)")
    print(f"  Post-filter: {pre_filter - len(train_data)} dropped (bad alternation in source data)")
    print(f"  TOTAL: {len(train_data)} training examples")

    # --- Generate full view for EVAL trajectories ---
    eval_data = []
    for ex in eval_examples:
        msgs = normalize_messages(ex["messages"])
        full = ensure_valid_alternation(msgs)
        if full and passes_converter_validation(full):
            eval_data.append({"messages": full})

    print(f"\n=== Eval: {len(eval_data)} full trajectories (from {len(eval_examples)} held-out tasks) ===")

    # --- Token statistics ---
    train_tokens = [estimate_tokens(d["messages"]) for d in train_data]
    eval_tokens = [estimate_tokens(d["messages"]) for d in eval_data]
    print(f"\n  Train tokens: min={min(train_tokens):,}, median={sorted(train_tokens)[len(train_tokens)//2]:,}, max={max(train_tokens):,}")
    for cutoff in [16384, 32768]:
        fits = sum(1 for t in train_tokens if t <= cutoff)
        print(f"  Fit in {cutoff//1024}K: {fits}/{len(train_tokens)} ({100*fits//len(train_tokens)}%)")
    print(f"  Eval tokens: min={min(eval_tokens):,}, median={sorted(eval_tokens)[len(eval_tokens)//2]:,}, max={max(eval_tokens):,}")

    # --- Write output ---
    print()
    write_jsonl(train_data, OUTPUT_DIR / "amdpilot_v4_train.jsonl")
    write_jsonl(eval_data, OUTPUT_DIR / "amdpilot_v4_eval.jsonl")


if __name__ == "__main__":
    main()
