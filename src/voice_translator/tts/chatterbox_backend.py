"""Chatterbox TTS benchmark adapter."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional
import numpy as np

from voice_translator.core.errors import ModelNotFoundError, WarmupError
from voice_translator.tts.base import TTSAdapter, VoiceProfile

logger = logging.getLogger(__name__)


class ChatterboxTTSAdapter(TTSAdapter):
    """Chatterbox Voice Cloning Adapter for benchmark comparison."""

    def __init__(self):
        self.model = None
        self.model_path = ""
        self.device = "cuda"
        self.sample_rate = 24000
        self._is_warm = False

    def initialize(self, model_path: str, device: str = "cuda", sample_rate: int = 24000):
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate

        p = Path(model_path)
        if not p.exists():
            raise ModelNotFoundError(
                f"Chatterbox model path '{model_path}' not found."
            )
        logger.info(f"Chatterbox adapter initialized with model '{model_path}'.")

    def warmup(self):
        self._is_warm = True

    def prepare_voice_profile(self, profile: VoiceProfile):
        pass

    def synthesize_committed(
        self,
        text: str,
        profile: VoiceProfile,
        target_language: str = "en",
    ) -> Iterator[np.ndarray]:
        # Benchmark stub
        if not text.strip():
            return
        # Emit dummy tone/silence for benchmark harness
        yield np.zeros(self.sample_rate, dtype=np.float32)

    def shutdown(self):
        self.model = None
        self._is_warm = False

