import json
from pathlib import Path
from typing import List, Sequence, Tuple


def read_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def split_dataset(records: Sequence[dict], base_samples: int, eval_samples: int) -> Tuple[List[dict], List[dict]]:
    total_needed = base_samples + eval_samples
    if len(records) < total_needed:
        raise ValueError(f"Dataset has {len(records)} records but {total_needed} are required.")
    base = list(records[:base_samples])
    evaluation = list(records[base_samples:base_samples + eval_samples])
    return base, evaluation
