import json
import subprocess
import sys
from pathlib import Path


def run_worker(tmp_path: Path, request: dict) -> tuple[int, list[dict]]:
    req = tmp_path / "request.json"
    req.write_text(json.dumps(request), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "shu2al_worker", str(req)], capture_output=True, text=True
    )
    return proc.returncode, [json.loads(line) for line in proc.stdout.splitlines()]


def test_mock_stage_reports_progress_then_result(tmp_path):
    code, events = run_worker(
        tmp_path,
        {"protocol": 1, "stage": "mock", "run_id": "r1", "output_dir": str(tmp_path / "out")},
    )
    assert code == 0
    assert [e["type"] for e in events] == ["progress"] * 5 + ["result"]
    assert events[-2]["pct"] == 100
    transcript = json.loads(Path(events[-1]["outputs"]["transcript"]).read_text(encoding="utf-8"))
    assert transcript["segments"][0]["text"] == "شو بدك تعمل هلق؟"


def test_unknown_stage_is_a_bad_request(tmp_path):
    code, events = run_worker(
        tmp_path, {"protocol": 1, "stage": "nope", "run_id": "r1", "output_dir": str(tmp_path)}
    )
    assert code == 1
    assert events == [{"type": "error", "code": "bad_request", "message": "unknown stage 'nope'"}]


def test_wrong_protocol_version(tmp_path):
    code, events = run_worker(
        tmp_path, {"protocol": 99, "stage": "mock", "run_id": "r1", "output_dir": str(tmp_path)}
    )
    assert code == 1
    assert events[0]["code"] == "bad_request"
