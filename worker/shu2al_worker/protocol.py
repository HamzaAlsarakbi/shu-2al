"""Worker protocol v1.

The Rust side writes a request envelope to a JSON file and runs
`python -m shu2al_worker <request.json>`. The worker answers on stdout with
JSON Lines: any number of `progress` events, then exactly one `result` or
`error`. Logs and tracebacks go to stderr.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL = 1


class StageError(Exception):
    """A failure with a machine-readable code the orchestrator can act on."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Request:
    stage: str
    run_id: str
    inputs: dict[str, str]
    output_dir: Path
    device: str = "auto"
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Request:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("protocol") != PROTOCOL:
            raise StageError("bad_request", f"unsupported protocol {data.get('protocol')!r}")
        try:
            return cls(
                stage=data["stage"],
                run_id=data["run_id"],
                inputs=data.get("inputs", {}),
                output_dir=Path(data["output_dir"]),
                device=data.get("device", "auto"),
                params=data.get("params", {}),
            )
        except KeyError as e:
            raise StageError("bad_request", f"missing field {e}") from e

    def input(self, name: str) -> Path:
        if name not in self.inputs:
            raise StageError("bad_request", f"missing input {name!r}")
        return Path(self.inputs[name])


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class Reporter:
    """Emits progress events, throttled so a tight loop doesn't flood the pipe."""

    def __init__(self, min_interval_s: float = 0.5):
        self._min_interval = min_interval_s
        self._last = 0.0

    def progress(self, pct: float, msg: str = "", *, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last >= self._min_interval:
            self._last = now
            emit({"type": "progress", "pct": round(max(0.0, min(100.0, pct)), 1), "msg": msg})


def write_json(path: Path, data: Any) -> None:
    """Writes atomically so a killed worker never leaves a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
