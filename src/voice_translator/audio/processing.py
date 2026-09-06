"""Offline WebRTC NS/AEC before microphone admission, outside capture callbacks."""

from __future__ import annotations

from collections import deque
import time

import numpy as np


class EchoReferenceBuffer:
    """Single callback producer, single DSP consumer; fixed one-second PCM history.

    Callback work is a bounded copy/append only. ASR never consumes this history,
    so slow incoming inference and paused subtitles cannot starve the reference.
    Timestamps describe the last captured sample, in the monotonic clock domain.
    """

    def __init__(self, sample_rate: int, frame_duration_ms: int):
        if frame_duration_ms <= 0 or sample_rate <= 0:
            raise ValueError("Echo reference requires positive sample rate and frame duration")
        self.sample_rate = sample_rate
        self._blocks = deque(maxlen=max(1, 1000 // frame_duration_ms))

    def push(self, pcm: np.ndarray, end_ns: int) -> None:
        self._blocks.append((end_ns, pcm.copy()))

    def frame(self, end_ns: int, count: int) -> tuple[np.ndarray, bool]:
        # Snapshot before inspecting: producer can continue appending independently.
        blocks = tuple(self._blocks)
        output = np.zeros(count, dtype=np.float32)
        covered = np.zeros(count, dtype=bool)
        start_ns = end_ns - round(count * 1e9 / self.sample_rate)
        for block_end, pcm in blocks:
            block_start = block_end - round(len(pcm) * 1e9 / self.sample_rate)
            offset = round((block_start - start_ns) * self.sample_rate / 1e9)
            left, right = max(0, offset), min(count, offset + len(pcm))
            if right > left:
                output[left:right] = pcm[left - offset:right - offset]
                covered[left:right] = True
        return output, bool(np.mean(covered) >= 0.9)

    def clear(self):
        self._blocks.clear()


class MicrophoneProcessor:
    """10 ms PCM16 WebRTC blocks; no model downloads, GPU allocation or ducking."""

    def __init__(self, config, reference: EchoReferenceBuffer | None = None, factory=None):
        self.config = config
        self.sample_rate = config.sample_rate
        if self.sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC microphone processing requires 8000, 16000, 32000 or 48000 Hz")
        if config.frame_duration_ms <= 0 or config.frame_duration_ms % 10:
            raise ValueError("WebRTC capture frame_duration_ms must be a positive multiple of 10")
        if factory is None:
            try:
                from aec_audio_processing import AudioProcessor
                factory = AudioProcessor
            except ImportError as exc:
                raise RuntimeError("Microphone processing requires aec-audio-processing==1.0.1; run uv sync") from exc
        self._factory = factory
        self.reference = reference
        self.block_size = self.sample_rate // 100
        self._pending = np.empty(0, dtype=np.float32)
        self._last_end_ns = None
        self._durations = deque(maxlen=1000)
        self._reference_health = deque(maxlen=100)
        self.frames = self.reference_misses = self.resets = 0
        self.last_error = None
        self._create_engine()

    def _create_engine(self):
        self.engine = self._factory(enable_aec=self.config.echo_cancellation,
                                    enable_ns=self.config.noise_suppression,
                                    ns_level=self.config.noise_suppression_level,
                                    enable_agc=False, enable_vad=False)
        self.engine.set_stream_format(self.sample_rate, 1)
        self.engine.set_reverse_stream_format(self.sample_rate, 1)
        # Leave one 10ms block of causal reference lead for WebRTC's filter bank.
        self.engine.set_stream_delay(min(10, self.config.echo_delay_ms))

    @staticmethod
    def _pcm16(pcm):
        return np.clip(np.rint(pcm * 32768), -32768, 32767).astype("<i2").tobytes()

    def reset(self):
        self._pending = np.empty(0, dtype=np.float32)
        self._last_end_ns = None
        self._create_engine()
        self.resets += 1

    def process(self, pcm: np.ndarray, end_ns: int) -> list[tuple[np.ndarray, int]]:
        if not len(pcm):
            return []
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("Microphone PCM must be finite mono audio")
        # Discontinuities invalidate the adaptive filter and partial block; never
        # join old PTT audio/reference to a new capture epoch after overload.
        start_ns = end_ns - round(len(pcm) * 1e9 / self.sample_rate)
        if self._last_end_ns is not None and abs(start_ns - self._last_end_ns) > 30_000_000:
            self.reset()
        self._last_end_ns = end_ns
        data = np.concatenate((self._pending, pcm))
        first_ns = end_ns - round(len(data) * 1e9 / self.sample_rate)
        outputs = []
        consumed = 0
        try:
            while len(data) - consumed >= self.block_size:
                block_end = first_ns + round((consumed + self.block_size) * 1e9 / self.sample_rate)
                block = data[consumed:consumed + self.block_size]
                started = time.perf_counter_ns()
                if self.config.echo_cancellation:
                    history_ms = max(0, self.config.echo_delay_ms - 10)
                    ref, complete = (self.reference.frame(block_end - history_ms * 1_000_000, self.block_size)
                                     if self.reference is not None else (np.zeros_like(block), False))
                    self.reference_misses += int(not complete)
                    self._reference_health.append(complete)
                    self.engine.process_reverse_stream(self._pcm16(ref))
                    self.engine.set_stream_delay(min(10, self.config.echo_delay_ms))
                result = self.engine.process_stream(self._pcm16(block))
                filtered = np.frombuffer(result, dtype="<i2").astype(np.float32) / 32768
                if len(filtered) != self.block_size:
                    raise RuntimeError("WebRTC returned an invalid frame size")
                outputs.append((filtered, block_end))
                consumed += self.block_size
                self.frames += 1
                self._durations.append((time.perf_counter_ns() - started) / 1e6)
        except Exception as exc:
            self.last_error = str(exc)
            raise
        self._pending = data[consumed:].copy()
        return outputs

    def snapshot(self) -> dict:
        p50, p95 = np.percentile(tuple(self._durations), [50, 95]) if self._durations else (None, None)
        return dict(backend="webrtc", noise_suppression=self.config.noise_suppression,
                    echo_cancellation=self.config.echo_cancellation,
                    noise_suppression_level=self.config.noise_suppression_level,
                    echo_delay_ms=self.config.echo_delay_ms, frames=self.frames,
                    reference_missing_frames=self.reference_misses, resets=self.resets,
                    reference_available=(sum(self._reference_health) / len(self._reference_health) >= 0.8
                                         if self._reference_health else False),
                    pending_samples=len(self._pending), p50_ms=p50, p95_ms=p95,
                    last_error=self.last_error)
