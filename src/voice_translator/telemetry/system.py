"""System resource monitor for GPU/VRAM and CPU usage."""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class SystemResourceMonitor:
    """Monitors CPU, System RAM, and GPU VRAM stats."""

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "cpu_percent": 0.0,
            "ram_used_mb": 0.0,
            "ram_total_mb": 0.0,
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_total_mb": 16384.0,  # 16 GB default for RTX 5060 Ti
            "gpu_available": False,
        }

        if psutil is not None:
            try:
                stats["cpu_percent"] = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                stats["ram_used_mb"] = round((mem.total - mem.available) / (1024 * 1024), 1)
                stats["ram_total_mb"] = round(mem.total / (1024 * 1024), 1)
            except Exception:
                pass

        if torch is not None and torch.cuda.is_available():
            try:
                stats["gpu_available"] = True
                stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
                stats["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 1)
                dev_prop = torch.cuda.get_device_properties(0)
                stats["gpu_total_mb"] = round(dev_prop.total_memory / (1024 * 1024), 1)
                stats["gpu_name"] = dev_prop.name
            except Exception:
                pass

        return stats

