"""Non-blocking WASAPI audio capture engine for mic and loopback streams."""

from __future__ import annotations

import logging
import time
from typing import Optional
import numpy as np

from voice_translator.audio.devices import AudioDeviceManager, DeviceInfo
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


class AudioCaptureEngine:
    """Captures WASAPI audio streams into a lockless PCM ring buffer with mono downmixing."""

    def __init__(
        self,
        device_info: DeviceInfo,
        sample_rate: int = 48000,
        channels: Optional[int] = None,
        frame_duration_ms: int = 20,
        ring_buffer_sec: float = 5.0,
    ):
        self.device_info = device_info
        self.sample_rate = sample_rate
        
        # Determine hardware channel count (WASAPI loopback is typically 2 channels)
        if channels is not None:
            self.channels = channels
        elif self.device_info.is_loopback:
            self.channels = max(1, self.device_info.max_input_channels)
        else:
            self.channels = max(1, min(2, self.device_info.max_input_channels)) if self.device_info.max_input_channels > 0 else 1

        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.sample_rate * (self.frame_duration_ms / 1000.0))
        capacity_samples = int(self.sample_rate * ring_buffer_sec)
        # Ring buffer stores single-channel mono PCM
        self.ring_buffer = PCMRingBuffer(capacity_samples=capacity_samples, channels=1, dtype=np.float32)

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self.is_running = False
        self.discontinuity_count = 0
        self.last_callback_ns = 0
        self.last_status_flags = 0
        self.last_error: Optional[str] = None
        self.signal_meter = AudioSignalMeter()
        self.callback_count = 0
        self.captured_samples = 0
        self.echo_reference = None

    def start(self):
        if self.is_running:
            return

        if pyaudio is None:
            raise AudioDeviceError("PyAudio/PyAudioWPatch is not installed.")

        self._pa = pyaudio.PyAudio()
        self.ring_buffer.clear()

        def _pyaudio_callback(in_data, frame_count, time_info, status_flags):
            self.callback_count += 1
            self.captured_samples += int(frame_count)
            self.last_callback_ns = time.monotonic_ns()
            self.last_status_flags = int(status_flags or 0)
            if status_flags:
                self.discontinuity_count += 1
            mono_audio = downmix_to_mono(np.frombuffer(in_data, dtype=np.int16), self.channels)
            self.ring_buffer.write(mono_audio)
            if self.echo_reference is not None:
                self.echo_reference.push(mono_audio, self.last_callback_ns)
            return (None, pyaudio.paContinue)

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_info.index,
                frames_per_buffer=self.frame_size,
                stream_callback=_pyaudio_callback,
            )
            self._stream.start_stream()
            self.is_running = True
            logger.info(f"Audio capture started on '{self.device_info.name}' (channels: {self.channels}, {self.sample_rate}Hz)")
        except Exception as e:
            self.last_error = str(e)
            self.stop()
            raise AudioDeviceError(f"Failed to open audio capture stream on '{self.device_info.name}': {e}") from e

    def read_samples(self, count: Optional[int] = None) -> np.ndarray:
        samples = self.ring_buffer.read(count)
        if len(samples):
            self.signal_meter.observe(samples)
        return samples

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
            "consumer_empty_reads": self.ring_buffer.underrun_count,
            "discontinuities": self.discontinuity_count,
            "callback_count": self.callback_count,
            "captured_samples": self.captured_samples,
            "last_callback_ns": self.last_callback_ns,
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
        logger.info(f"Audio capture stopped for device: {self.device_info.name}")
