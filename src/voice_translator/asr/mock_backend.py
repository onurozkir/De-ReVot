"""Deterministic mock ASR adapter for testing."""

from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

from voice_translator.asr.base import ASRAdapter, ASRSession
from voice_translator.core.types import Direction, UtteranceEvent, UtteranceState


class MockASRAdapter(ASRAdapter):
    """Fake ASR adapter that returns configurable canned sentences."""

    def __init__(self, predefined_transcripts: Optional[List[str]] = None):
        self.transcripts = predefined_transcripts or ["Merhaba nasılsınız", "Toplantıya hoş geldiniz", "Bu bir test konuşmasıdır"]
        self._idx = 0
        self.is_initialized = False
        self.is_warmed = False

    def initialize(self, model_path: str = "", device: str = "cpu", compute_type: str = "float32"):
        self.is_initialized = True

    def warmup(self):
        self.is_warmed = True

    def create_session(self, stream_id: str, direction: Direction, language: str) -> ASRSession:
        return ASRSession(
            stream_id=stream_id,
            direction=direction,
            language=language,
        )

    def process_audio(
        self,
        session: ASRSession,
        audio_chunk_16k: np.ndarray,
        captured_at_ns: int,
    ) -> Optional[UtteranceEvent]:
        if not session.is_active or len(audio_chunk_16k) == 0:
            return None

        session.total_audio_samples += len(audio_chunk_16k)
        if session.total_audio_samples < 8000:
            return None

        text = self.transcripts[self._idx % len(self.transcripts)]
        session.current_revision += 1
        session.last_partial_text = text

        return UtteranceEvent(
            meeting_id="test_meeting",
            stream_id=session.stream_id,
            direction=session.direction,
            utterance_id=f"{session.stream_id}_{session.sequence_id}",
            sequence_id=session.sequence_id,
            revision=session.current_revision,
            state=UtteranceState.PARTIAL,
            source_language=session.language,
            text=text,
            audio_start_ns=captured_at_ns,
            audio_end_ns=captured_at_ns,
            is_final=False,
            model_info={"backend": "mock_asr"},
        )

    def flush_session(self, session: ASRSession) -> Optional[UtteranceEvent]:
        text = session.last_partial_text or self.transcripts[self._idx % len(self.transcripts)]
        self._idx += 1
        seq_id = session.sequence_id
        session.sequence_id += 1
        session.current_revision = 0
        session.total_audio_samples = 0
        session.last_partial_text = ""

        now_ns = time.monotonic_ns()
        return UtteranceEvent(
            meeting_id="test_meeting",
            stream_id=session.stream_id,
            direction=session.direction,
            utterance_id=f"{session.stream_id}_{seq_id}",
            sequence_id=seq_id,
            revision=1,
            state=UtteranceState.COMMITTED,
            source_language=session.language,
            text=text,
            audio_start_ns=now_ns,
            audio_end_ns=now_ns,
            is_final=True,
            model_info={"backend": "mock_asr"},
        )

    def close_session(self, session: ASRSession):
        session.is_active = False

    def shutdown(self):
        self.is_initialized = False
        self.is_warmed = False

