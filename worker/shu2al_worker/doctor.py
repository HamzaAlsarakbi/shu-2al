"""`doctor` stage: report the GPU and library versions the worker sees."""

from __future__ import annotations

import platform

from .protocol import Reporter, Request


def run(req: Request, report: Reporter) -> dict:
    info: dict = {"python": platform.python_version()}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return {"stats": info}
    info["torch"] = torch.__version__
    info["cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_free_mb"] = free // 2**20
        info["vram_total_mb"] = total // 2**20
        info["bf16"] = torch.cuda.is_bf16_supported()
    for lib in ("transformers", "silero_vad", "ctc_forced_aligner"):
        try:
            mod = __import__(lib)
            info[lib] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[lib] = None
    return {"stats": info}
