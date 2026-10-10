"""Local, append-only diagnostics for reproducible RF-Gensec training."""

import json
from pathlib import Path


def append_jsonl(path, record):
    """Append one JSON object and flush it so interrupted runs keep prior updates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()

