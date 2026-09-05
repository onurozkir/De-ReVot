"""Core data types and contracts across the entire translator pipeline."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


class Direction(str, enum.Enum):
    OUTGOING = "outgoing"  # TR mic -> EN speech -> VB-CABLE -> Teams
    INCOMING = "incoming"  # EN Teams loopback -> TR text -> Web UI


class UtteranceState(str, enum.Enum):
    IDLE = "idle"
    SPEECH_DETECTED = "speech_detected"
    PARTIAL = "partial"
    COMMIT_CANDIDATE = "commit_candidate"
    COMMITTED = "committed"
    DISPATCHED = "dispatched"
    DELIVERED = "delivered"
    CLOSED = "closed"


class MeetingStatus(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    WARMING = "warming"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass(slots=True)
class AudioFrame:
    """PCM audio frame captured from WASAPI callback."""
    data: np.ndarray  # float32 or int16 1D/2D array
    sample_rate: int
    channels: int
    captured_at_ns: int = field(default_factory=time.monotonic_ns)

    @property
    def duration_ms(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return (len(self.data) / self.sample_rate) * 1000.0


@dataclass(slots=True)
class UtteranceEvent:
    """ASR hypothesis or committed utterance event."""
    meeting_id: str
    stream_id: str
    direction: Direction
    utterance_id: str
    sequence_id: int
    revision: int
    state: UtteranceState
    source_language: str
    text: str
    audio_start_ns: int
    audio_end_ns: int
    created_at_ns: int = field(default_factory=time.monotonic_ns)
    is_final: bool = False
    model_info: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationEvent:
    """Translated text event."""
    meeting_id: str
    utterance_id: str
    sequence_id: int
    revision: int
    direction: Direction
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    state: UtteranceState
    created_at_ns: int = field(default_factory=time.monotonic_ns)
    model_info: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TTSChunkEvent:
    """Synthesized PCM audio chunk from TTS."""
    meeting_id: str
    utterance_id: str
    sequence_id: int
    chunk_index: int
    pcm_data: np.ndarray  # float32 or int16 1D mono
    sample_rate: int
    is_final: bool = False
    created_at_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(slots=True)
class LatencyEvent:
    """Monotonic latency tracking sample."""
    meeting_id: str
    utterance_id: Optional[str]
    direction: Direction
    event_type: str
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    duration_ms: Optional[float] = None
    queue_age_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
