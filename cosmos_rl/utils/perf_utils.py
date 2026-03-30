import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import DefaultDict, Dict, Iterable

import torch


def stage_perf_enabled() -> bool:
    value = os.getenv("COSMOS_STAGE_PERF", "0").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def new_perf_metrics() -> DefaultDict[str, float]:
    return defaultdict(float)


def _sync_cuda(cuda_device) -> None:
    if cuda_device is None:
        return
    if isinstance(cuda_device, torch.device):
        if cuda_device.type != "cuda":
            return
        torch.cuda.synchronize(cuda_device)
        return
    if isinstance(cuda_device, str):
        if not cuda_device.startswith("cuda"):
            return
        torch.cuda.synchronize(cuda_device)
        return
    torch.cuda.synchronize(cuda_device)


@contextmanager
def measure_time(
    metrics: DefaultDict[str, float],
    key: str,
    *,
    enabled: bool,
    cuda_device=None,
):
    if not enabled:
        yield
        return

    _sync_cuda(cuda_device)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        _sync_cuda(cuda_device)
        metrics[key] += time.perf_counter() - start_time


def perf_metrics_to_dict(metrics: DefaultDict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def inject_perf_metrics(
    target: Dict[str, float],
    metrics: DefaultDict[str, float],
    *,
    prefix: str,
) -> None:
    for key, value in metrics.items():
        target[f"{prefix}/{key}"] = float(value)


def format_perf_metrics(
    metrics: Dict[str, float] | DefaultDict[str, float],
    *,
    keys: Iterable[str] | None = None,
) -> str:
    if keys is None:
        keys = sorted(metrics.keys())
    return ", ".join(f"{key}={float(metrics[key]):.4f}s" for key in keys if key in metrics)
