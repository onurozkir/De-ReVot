"""Blocking hardware diagnostics; never imported by realtime callbacks."""

from __future__ import annotations

import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from voice_translator.audio.capture import AudioCaptureEngine
from voice_translator.audio.devices import DeviceInfo
from voice_translator.audio.signal import signal_levels


def frame_level_summary(
    pcm: np.ndarray,
    sample_rate: int = 48000,
    frame_duration_ms: int = 20,
    active_rms_threshold: float = 0.001,
) -> dict:
    frame_size = max(1, int(sample_rate * frame_duration_ms / 1000))
    values = np.asarray(pcm, dtype=np.float32).reshape(-1)
    frame_rms = np.asarray(
        [
            float(np.sqrt(np.mean(values[offset : offset + frame_size] ** 2)))
            for offset in range(0, len(values) - frame_size + 1, frame_size)
        ],
        dtype=np.float32,
    )
    active_frames = int(np.count_nonzero(frame_rms >= active_rms_threshold))
    return {
        "frame_duration_ms": frame_duration_ms,
        "frame_count": int(len(frame_rms)),
        "max_frame_rms": float(np.max(frame_rms)) if len(frame_rms) else 0.0,
        "p95_frame_rms": float(np.percentile(frame_rms, 95)) if len(frame_rms) else 0.0,
        "active_rms_threshold": active_rms_threshold,
        "active_frames": active_frames,
        "active_duration_ms": active_frames * frame_duration_ms,
    }


def dominant_frequency(pcm: np.ndarray, sample_rate: int = 48000) -> float:
    values = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if len(values) < sample_rate // 10:
        return 0.0
    spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values))))
    return float(np.fft.rfftfreq(len(values), 1.0 / sample_rate)[int(np.argmax(spectrum))])


def capture_endpoint(
    device: DeviceInfo,
    duration_sec: float,
    sample_rate: int = 48000,
    on_started: Optional[Callable[[], None]] = None,
) -> tuple[np.ndarray, dict]:
    engine = AudioCaptureEngine(device_info=device, sample_rate=sample_rate, ring_buffer_sec=max(2.0, duration_sec + 1.0))
    chunks: list[np.ndarray] = []
    engine.start()
    if on_started is not None:
        on_started()
    deadline = time.monotonic() + duration_sec
    try:
        while time.monotonic() < deadline:
            samples = engine.read_samples(engine.frame_size)
            if len(samples):
                chunks.append(samples)
            time.sleep(0.005)
        tail = engine.read_samples()
        if len(tail):
            chunks.append(tail)
    finally:
        engine.stop()
    pcm = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    rms, peak, dbfs = signal_levels(pcm)
    diagnostics = engine.get_diagnostics()
    diagnostics["recording"] = {
        "frames": len(chunks), "samples": len(pcm), "duration_sec": len(pcm) / sample_rate,
        "rms": rms, "peak": peak, "dbfs": dbfs,
        "frame_levels": frame_level_summary(pcm, sample_rate),
    }
    return pcm, diagnostics


def write_pcm16_wav(path: Path, pcm: np.ndarray, sample_rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    int16 = (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.tobytes())
