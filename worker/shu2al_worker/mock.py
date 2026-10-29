"""`mock` stage: replays a fixture transcript with fake progress, so the server,
queue and editor can be developed without a GPU."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .protocol import Reporter, Request, write_json

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "transcript.json"


def run(req: Request, report: Reporter) -> dict:
    fixture = Path(req.params.get("fixture", FIXTURE))
    steps = int(req.params.get("steps", 5))
    for i in range(steps):
        report.progress(100 * (i + 1) / steps, f"mock step {i + 1}/{steps}", force=True)
        time.sleep(float(req.params.get("step_s", 0.0)))
    out_path = req.output_dir / "transcript.json"
    write_json(out_path, json.loads(fixture.read_text(encoding="utf-8")))
    return {"outputs": {"transcript": str(out_path)}, "stats": {"mock": True}}
