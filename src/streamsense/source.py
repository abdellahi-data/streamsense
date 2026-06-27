"""Replay a dataset as a stream of events."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from streamsense.schemas import Event


def replay(path: str | Path, rate_per_sec: float = 10.0, loop: bool = False) -> Iterator[Event]:
    """Yield events from a JSONL file, paced to simulate a live feed."""
    delay = 1.0 / rate_per_sec
    while True:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            yield Event(text=row["text"], label=row.get("label"), source="replay")
            time.sleep(delay)
        if not loop:
            break