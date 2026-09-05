"""Audio playback and rendering engine for output devices (e.g. VB-CABLE Input)."""

from __future__ import annotations

import logging
import time
from typing import Optional
import numpy as np

from voice_translator.audio.devices import DeviceInfo
from voice_translator.audio.resampler import AudioResampler
from voice_translator.audio.signal import AudioSignalMeter, downmix_to_mono
from voice_translator.core.errors import AudioDeviceError
from voice_translator.core.ring_buffer import PCMRingBuffer

logger = logging.getLogger(__name__)

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    try:
        import pyaudio
    except ImportError:
        pyaudio = None  # type: ignore


class AudioRenderEngine:
    """Renders PCM audio to an output endpoint (such as VB-CABLE Input) with stereo duplication."""

    def __init__(
        self,
        device_info: DeviceInfo,
        sample_rate: int = 48000,
        channels: Optional[int] = None,
        frame_duration_ms: int = 20,
        ring_buffer_sec: float = 10.0,
    ):
        self.device_info = device_info
        self.sample_rate = sample_rate
        
        # Native output channels (VB-CABLE is 2 channels)
        if channels is not None:
            self.channels = channels
        else:
            self.channels = max(1, min(2, self.device_info.max_output_channels)) if self.device_info.max_output_channels > 0 else 2

        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))

        capacity_samples = int(self.sample_rate * ring_buffer_sec)
        # Ring buffer stores single-channel mono PCM
        self.ring_buffer = PCMRingBuffer(capacity_samples=capacity_samples, channels=1, dtype=np.float32)

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self.is_running = False
        self.discontinuity_count = 0
        self.last_status_flags = 0
        self.last_callback_ns = 0
        self.last_write_ns = 0
        self.last_error: Optional[str] = None
        self.signal_meter = AudioSignalMeter()
        self._resamplers: dict[int, AudioResampler] = {}
        self.muted = False

    def start(self):
        if self.is_running:
            return

        if pyaudio is None:
            raise AudioDeviceError("PyAudio/PyAudioWPatch is not installed.")

        self._pa = pyaudio.PyAudio()
        self.ring_buffer.clear()

        def _pyaudio_callback(in_data, frame_count, time_info, status_flags):
            self.last_callback_ns = time.monotonic_ns()
            self.last_status_flags = int(status_flags or 0)
            if status_flags:
                self.discontinuity_count += 1
            # Read mono samples from ring buffer
            samples = np.empty(0, dtype=np.float32) if self.muted else self.ring_buffer.read(frame_count)
            if len(samples) < frame_count:
                pad = np.zeros(frame_count - len(samples), dtype=np.float32)
                samples = np.concatenate([samples, pad]) if len(samples) > 0 else pad

            # If device expects 2 channels (stereo), duplicate mono to Left/Right
            if self.channels == 2:
                stereo_samples = np.column_stack([samples, samples]).ravel()
                int16_samples = (np.clip(stereo_samples, -1.0, 1.0) * 32767.0).astype(np.int16)
            else:
                int16_samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)

            return (int16_samples.tobytes(), pyaudio.paContinue)

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                output_device_index=self.device_info.index,
                frames_per_buffer=self.frame_size,
                stream_callback=_pyaudio_callback,
            )
            self._stream.start_stream()
            self.is_running = True
            logger.info(f"Audio render started on '{self.device_info.name}' (channels: {self.channels}, {self.sample_rate}Hz)")
        except Exception as e:
            self.last_error = str(e)
            self.stop()
            raise AudioDeviceError(f"Failed to open audio render stream on '{self.device_info.name}': {e}") from e

    def push_pcm(self, pcm_data: np.ndarray, source_rate: Optional[int] = None):
        """Write PCM audio to the render buffer with automatic resampling if needed."""
        if self.muted or len(pcm_data) == 0:
            return

        if source_rate is not None and source_rate != self.sample_rate:
            resampler = self._resamplers.get(source_rate)
            if resampler is None:
                resampler = AudioResampler(
                    in_rate=source_rate, out_rate=self.sample_rate,
                    in_channels=1, out_channels=1, streaming=True,
                )
                self._resamplers[source_rate] = resampler
            pcm_data = resampler.process(pcm_data)

        pcm_data = downmix_to_mono(pcm_data)
        self.signal_meter.observe(pcm_data)
        self.ring_buffer.write(pcm_data)
        self.last_write_ns = time.monotonic_ns()

    def flush_source(self, source_rate: int) -> None:
        if self.muted:
            return
        resampler = self._resamplers.get(source_rate)
        if resampler is None:
            return
        tail = downmix_to_mono(resampler.flush())
        if len(tail):
            self.signal_meter.observe(tail)
            self.ring_buffer.write(tail)
            self.last_write_ns = time.monotonic_ns()

    def set_muted(self, muted: bool) -> None:
        self.muted = muted
        if muted:
            self.ring_buffer.clear()
            self._resamplers.clear()

    def get_diagnostics(self) -> dict:
        return {
            "device": self.device_info.to_dict(),
            "stream_state": "running" if self.is_running else "stopped",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frame_size": self.frame_size,
            "signal": self.signal_meter.snapshot(),
            "queue_depth_samples": self.ring_buffer.available_read,
            "queue_age_ms": self.ring_buffer.available_read / self.sample_rate * 1000.0,
            "queue_capacity_samples": self.ring_buffer.capacity,
            "overruns": self.ring_buffer.overrun_count,
            "underruns": self.ring_buffer.underrun_count,
            "discontinuities": self.discontinuity_count,
            "last_callback_ns": self.last_callback_ns,
            "last_write_ns": self.last_write_ns,
            "last_status_flags": self.last_status_flags,
            "last_error": self.last_error,
        }

    def stop(self):
        self.is_running = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        self._resamplers.clear()
        logger.info(f"Audio render stopped for device: {self.device_info.name}")
