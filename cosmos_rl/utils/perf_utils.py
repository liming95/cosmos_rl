import os
import time
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
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


def accumulate_perf_metrics(
    total_metrics: DefaultDict[str, float],
    step_metrics: Dict[str, float] | DefaultDict[str, float],
) -> None:
    for key, value in step_metrics.items():
        total_metrics[key] += float(value)


def accumulate_perf_metric_counts(
    total_counts: DefaultDict[str, int],
    step_metrics: Dict[str, float] | DefaultDict[str, float],
) -> None:
    for key in step_metrics.keys():
        total_counts[key] += 1


def inject_perf_summary(
    target: Dict[str, float],
    total_metrics: Dict[str, float] | DefaultDict[str, float],
    *,
    prefix: str,
    count: int,
    metric_counts: Dict[str, int] | DefaultDict[str, int] | None = None,
) -> None:
    count = max(int(count), 1)
    target[f"{prefix}/count"] = float(count)
    for key, value in total_metrics.items():
        value = float(value)
        per_metric_count = (
            max(int(metric_counts.get(key, 0)), 1)
            if metric_counts is not None
            else count
        )
        target[f"{prefix}/metric_count/{key}"] = float(per_metric_count)
        target[f"{prefix}/total/{key}"] = value
        target[f"{prefix}/avg/{key}"] = value / per_metric_count


def summarize_perf_metrics(
    total_metrics: Dict[str, float] | DefaultDict[str, float],
    *,
    count: int,
    metric_counts: Dict[str, int] | DefaultDict[str, int] | None = None,
) -> Dict[str, float]:
    count = max(int(count), 1)
    summary: Dict[str, float] = {"count": float(count)}
    for key, value in total_metrics.items():
        value = float(value)
        per_metric_count = (
            max(int(metric_counts.get(key, 0)), 1)
            if metric_counts is not None
            else count
        )
        summary[f"metric_count/{key}"] = float(per_metric_count)
        summary[f"total/{key}"] = value
        summary[f"avg/{key}"] = value / per_metric_count
    return summary


def summarize_perf_metrics_by_prefix(
    total_metrics: Dict[str, float] | DefaultDict[str, float],
    *,
    prefix: str,
    count: int,
    metric_counts: Dict[str, int] | DefaultDict[str, int] | None = None,
) -> Dict[str, float]:
    filtered_metrics: Dict[str, float] = {
        key: float(value)
        for key, value in total_metrics.items()
        if key.startswith(prefix)
    }
    filtered_counts: Dict[str, int] | None = None
    if metric_counts is not None:
        filtered_counts = {
            key: int(value)
            for key, value in metric_counts.items()
            if key.startswith(prefix)
        }
    return summarize_perf_metrics(
        filtered_metrics,
        count=count,
        metric_counts=filtered_counts,
    )


def summarize_perf_metrics_by_keys(
    total_metrics: Dict[str, float] | DefaultDict[str, float],
    *,
    keys: Iterable[str],
    count: int,
    metric_counts: Dict[str, int] | DefaultDict[str, int] | None = None,
) -> Dict[str, float]:
    key_list = list(keys)
    filtered_metrics: Dict[str, float] = {
        key: float(total_metrics[key]) for key in key_list if key in total_metrics
    }
    filtered_counts: Dict[str, int] | None = None
    if metric_counts is not None:
        filtered_counts = {
            key: int(metric_counts[key]) for key in key_list if key in metric_counts
        }
    return summarize_perf_metrics(
        filtered_metrics,
        count=count,
        metric_counts=filtered_counts,
    )


def write_perf_summary(
    output_dir: str,
    *,
    category: str,
    replica_name: str,
    global_rank: int,
    payload: Dict,
) -> str:
    summary_dir = Path(output_dir) / "perf_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"{category}_{replica_name}_{global_rank}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(path)


def append_perf_summary(output_dir: str, payload: Dict) -> str:
    summary_dir = Path(output_dir) / "perf_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / "all_summaries.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
    return str(path)


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
