"""Mock TTS Adapter for tests and offline validation."""

from __future__ import annotations

from typing import Iterator
import numpy as np

from voice_translator.tts.base import TTSAdapter, VoiceProfile


class MockTTSAdapter(TTSAdapter):
    """Deterministic mock TTS generating simple sine wave chunks."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.is_initialized = False
        self.is_warmed = False

    def initialize(self, model_path: str = "", device: str = "cpu", sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.is_initialized = True

    def warmup(self):
        self.is_warmed = True

    def prepare_voice_profile(self, profile: VoiceProfile):
        pass

    def synthesize_committed(
        self,
        text: str,
        profile: VoiceProfile,
        target_language: str = "en",
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return

        # Generate 0.5s of 440Hz sine wave as mock synthesized audio
        duration_sec = min(3.0, max(0.4, len(text) * 0.05))
        total_samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        chunk_size = int(self.sample_rate * 0.2)  # 200ms chunks
        for i in range(0, len(audio), chunk_size):
            yield audio[i : i + chunk_size]

    def shutdown(self):
        self.is_initialized = False
        self.is_warmed = False

