"""Abstract Base Class and VoiceProfile for TTS / Voice Cloning."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import numpy as np


@dataclass
class VoiceProfile:
    """Voice profile manifest for voice cloning."""
    id: str
    display_name: str
    backend: str
    reference_audio_path: str = ""
    reference_audio_paths: List[str] = field(default_factory=list)
    reference_text: Optional[str] = None
    reference_language: str = "tr"
    target_language: str = "en"
    target_languages: List[str] = field(default_factory=lambda: ["en", "fr"])
    is_default: bool = False
    conditioning_cache_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reference_audio_path and self.reference_audio_paths:
            self.reference_audio_path = self.all_reference_paths[0]

    @property
    def all_reference_paths(self) -> List[str]:
        """Returns all reference audio paths, combining reference_audio_path and reference_audio_paths."""
        paths: List[str] = []
        if self.reference_audio_path:
            paths.append(self.reference_audio_path)
        for p in self.reference_audio_paths:
            if p and p not in paths:
                paths.append(p)
        return [str(p) for p in sorted({Path(os.path.abspath(p)) for p in paths})]


class TTSAdapter(abc.ABC):
    """Abstract contract for Voice Cloning TTS engines."""

    @abc.abstractmethod
    def initialize(self, model_path: str, device: str = "cuda", sample_rate: int = 24000):
        """Initialize TTS model offline; validate local paths."""
        pass

    @abc.abstractmethod
    def warmup(self):
        """Warm up model weights and CUDA kernels."""
        pass

    @abc.abstractmethod
    def prepare_voice_profile(self, profile: VoiceProfile):
        """Extract and cache speaker conditioning latents/embeddings."""
        pass

    @abc.abstractmethod
    def synthesize_committed(
        self,
        text: str,
        profile: VoiceProfile,
        target_language: str = "en",
    ) -> Iterator[np.ndarray]:
        """Synthesize ONLY committed text into 1D float32 mono PCM chunks."""
        pass

    @abc.abstractmethod
    def shutdown(self):
        """Release resources."""
        pass

