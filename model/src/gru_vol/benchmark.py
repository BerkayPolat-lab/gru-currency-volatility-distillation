"""Batch-1 inference latency (median and p95, CPU and GPU separately) and serialized model size."""

import io
import platform
import time

import numpy as np
import torch


def param_count(model):
    """Number of trainable weights."""
    return sum(p.numel() for p in model.parameters())


def serialized_size_kb(model):
    """Size of the saved state dict (weights only, no optimizer state) in KB (1 KB = 1024 bytes)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1024


def cpu_name():
    """CPU model for the latency caption, e.g. 'Intel(R) Xeon(R) CPU @ 2.20GHz' on Colab."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass
    return platform.processor() or platform.machine()


def cpu_latency_ms(forecast, x, cfg):
    """Median and p95 latency of forecast(x) in ms on the CPU, using the benchmark settings in the config.

    Warm up first, time each batch-1 call separately, and exclude any file loading.
    Wrap model calls in torch.no_grad() and pass a model in eval mode.
    """
    b = cfg["benchmark"]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(b["num_threads"])
    try:
        for _ in range(b["warmup_runs"]):
            forecast(x)
        times = []
        for _ in range(b["timed_runs"]):
            start = time.perf_counter()
            forecast(x)
            times.append((time.perf_counter() - start) * 1e3)
    finally:
        torch.set_num_threads(previous_threads)  # leave later training cells at full speed
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))
