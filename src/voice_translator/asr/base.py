"""Abstract Base Classes for ASR Adapters."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from voice_translator.core.types import Direction, UtteranceEvent, UtteranceState


@dataclass
class ASRSession:
    """Session state for an isolated audio stream."""
    stream_id: str
    direction: Direction
    language: str
    sequence_id: int = 0
    current_revision: int = 0
    audio_buffer: List[np.ndarray] = field(default_factory=list)
    total_audio_samples: int = 0
    last_partial_text: str = ""
    stable_prefix: str = ""
    prefix_agreement_count: int = 0
    is_active: bool = True
    initial_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ASRAdapter(abc.ABC):
    """Abstract contract for ASR backends."""

    @abc.abstractmethod
    def initialize(self, model_path: str, device: str = "cuda", compute_type: str = "float16"):
        """Initialize the model offline; validate local paths."""
        pass

    @abc.abstractmethod
    def warmup(self):
        """Warm up model weights with dummy inference."""
        pass

    @abc.abstractmethod
    def create_session(self, stream_id: str, direction: Direction, language: str) -> ASRSession:
        """Create an isolated session for an audio stream."""
        pass

    @abc.abstractmethod
    def process_audio(
        self,
        session: ASRSession,
        audio_chunk_16k: np.ndarray,
        captured_at_ns: int,
    ) -> Optional[UtteranceEvent]:
        """Feed 16kHz mono float32 audio and return hypothesis if available."""
        pass

    @abc.abstractmethod
    def flush_session(self, session: ASRSession) -> Optional[UtteranceEvent]:
        """Flush any remaining audio buffer and produce final committed utterance."""
        pass

    @abc.abstractmethod
    def close_session(self, session: ASRSession):
        """Clean up session state."""
        pass

    @abc.abstractmethod
    def shutdown(self):
        """Unload models and release resources."""
        pass

