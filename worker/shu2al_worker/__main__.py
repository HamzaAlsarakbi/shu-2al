"""`python -m shu2al_worker <request.json>`: run one stage and report as JSON Lines."""

from __future__ import annotations

import importlib
import sys
import traceback

from .protocol import Reporter, Request, StageError, emit

STAGES = {
    "asr": "shu2al_worker.asr",
    "align": "shu2al_worker.align",
    "doctor": "shu2al_worker.doctor",
    "mock": "shu2al_worker.mock",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m shu2al_worker <request.json>", file=sys.stderr)
        return 2
    try:
        req = Request.load(argv[0])
        if req.stage not in STAGES:
            raise StageError("bad_request", f"unknown stage {req.stage!r}")
        stage = importlib.import_module(STAGES[req.stage])
        result = stage.run(req, Reporter())
    except StageError as e:
        emit({"type": "error", "code": e.code, "message": str(e)})
        return 1
    except Exception as e:  # noqa: BLE001 - report every failure to the orchestrator
        traceback.print_exc(file=sys.stderr)
        oom = type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()
        # A game grabbing VRAM mid-run shows up as OOM; the orchestrator requeues these.
        emit({"type": "error", "code": "gpu_contention" if oom else "internal", "message": str(e)})
        return 1
    emit({"type": "result", **result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
