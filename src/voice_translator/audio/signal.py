"""PCM conversion and live signal diagnostics shared by capture and render paths."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, asdict

import numpy as np


def pcm_to_float32(audio: np.ndarray) -> np.ndarray:
    """Return float32 PCM in [-1, 1] without asymmetric int16 overflow."""
    values = np.asarray(audio)
    if np.issubdtype(values.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
        values = values.astype(np.float32) / scale
    elif np.issubdtype(values.dtype, np.unsignedinteger):
        info = np.iinfo(values.dtype)
        midpoint = (info.max + 1) / 2.0
        values = (values.astype(np.float32) - midpoint) / midpoint
    else:
        values = values.astype(np.float32, copy=False)
    return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)


def downmix_to_mono(audio: np.ndarray, channels: int | None = None) -> np.ndarray:
    """Downmix interleaved or matrix PCM to mono using an arithmetic mean."""
    values = pcm_to_float32(audio)
    if values.ndim == 2:
        return values.mean(axis=1, dtype=np.float32)
    if channels and channels > 1:
        usable = (len(values) // channels) * channels
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        return values[:usable].reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return values.reshape(-1)


def signal_levels(audio: np.ndarray) -> tuple[float, float, float]:
    """Return RMS, absolute peak, and dBFS for float-compatible PCM."""
    values = pcm_to_float32(audio).reshape(-1)
    if len(values) == 0:
        return 0.0, 0.0, -120.0
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    peak = float(np.max(np.abs(values)))
    dbfs = max(-120.0, 20.0 * math.log10(max(rms, 1e-6)))
    return rms, peak, dbfs


@dataclass(slots=True)
class SignalSnapshot:
    rms: float = 0.0
    peak: float = 0.0
    dbfs: float = -120.0
    frames_observed: int = 0
    samples_observed: int = 0
    last_observed_ns: int = 0


class AudioSignalMeter:
    """Thread-safe last-frame meter; observation occurs outside audio callbacks."""

    def __init__(self) -> None:
        self._snapshot = SignalSnapshot()
        self._lock = threading.Lock()

    def observe(self, audio: np.ndarray, observed_at_ns: int | None = None) -> SignalSnapshot:
        rms, peak, dbfs = signal_levels(audio)
        with self._lock:
            self._snapshot.rms = rms
            self._snapshot.peak = peak
            self._snapshot.dbfs = dbfs
            self._snapshot.frames_observed += 1
            self._snapshot.samples_observed += len(audio)
            self._snapshot.last_observed_ns = observed_at_ns or time.monotonic_ns()
            return SignalSnapshot(**asdict(self._snapshot))

    def snapshot(self) -> dict:
        with self._lock:
            data = asdict(self._snapshot)
        data["age_ms"] = (
            (time.monotonic_ns() - data["last_observed_ns"]) / 1e6
            if data["last_observed_ns"]
            else None
        )
        return data
