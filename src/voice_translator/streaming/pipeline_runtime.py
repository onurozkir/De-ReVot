"""Shared runtime construction and utterance reset helpers for both directions."""

from __future__ import annotations

from voice_translator.asr.base import ASRSession
from voice_translator.config.models import StreamingConfig
from voice_translator.streaming.hallucination_guard import (
    HallucinationGuard,
    HallucinationPolicy,
    SpeechEvidence,
)
from voice_translator.streaming.vad import SileroVAD, VADResult


_CARRIED_PARTIAL_TEXT = "carried_partial_text"
_CARRIED_AUDIO_START_NS = "carried_audio_start_ns"
_CARRIED_MODEL_INFO = "carried_model_info"


def build_vad(config: StreamingConfig) -> SileroVAD:
    return SileroVAD(
        threshold=config.vad_threshold,
        end_threshold=config.vad_end_threshold,
        min_speech_duration_ms=config.vad_min_speech_duration_ms,
        min_silence_duration_ms=config.vad_min_silence_duration_ms,
        start_confirm_frames=config.vad_start_confirm_frames,
        end_confirm_frames=config.vad_end_confirm_frames,
        hangover_ms=config.vad_hangover_ms,
        energy_start_dbfs=config.vad_energy_start_dbfs,
        energy_end_dbfs=config.vad_energy_end_dbfs,
    )


def build_guard(config: StreamingConfig) -> HallucinationGuard:
    return HallucinationGuard(
        HallucinationPolicy(
            min_utterance_ms=config.guard_min_utterance_ms,
            min_voiced_ms=config.guard_min_voiced_ms,
            min_voiced_ratio=config.guard_min_voiced_ratio,
            max_audio_queue_age_ms=config.max_audio_queue_age_ms,
            max_no_speech_prob=config.whisper_max_no_speech_prob,
            min_avg_logprob=config.whisper_min_avg_logprob,
            max_compression_ratio=config.whisper_max_compression_ratio,
            max_chars_per_second=config.guard_max_chars_per_second,
        )
    )


def speech_evidence(result: VADResult, max_queue_age_ms: float) -> SpeechEvidence:
    return SpeechEvidence(
        utterance_ms=result.utterance_ms,
        voiced_ms=result.voiced_ms,
        voiced_ratio=result.voiced_ratio,
        max_queue_age_ms=max_queue_age_ms,
    )


def reset_asr_utterance(
    session: ASRSession | None,
    *,
    carried_partial_text: str = "",
    carried_audio_start_ns: int | None = None,
    carried_model_info: dict | None = None,
) -> None:
    if session is None:
        return
    session.audio_buffer.clear()
    session.total_audio_samples = 0
    session.last_partial_text = carried_partial_text.strip()
    session.current_revision = 1 if session.last_partial_text else 0
    session.metadata.pop("last_model_info", None)
    session.metadata.pop("samples_since_decode", None)
    session.metadata.pop("decode_attempted", None)
    session.metadata.pop("capture_start_ns", None)
    session.metadata.pop("capture_end_ns", None)
    session.metadata.pop(_CARRIED_PARTIAL_TEXT, None)
    session.metadata.pop(_CARRIED_AUDIO_START_NS, None)
    session.metadata.pop(_CARRIED_MODEL_INFO, None)
    if session.last_partial_text:
        session.metadata[_CARRIED_PARTIAL_TEXT] = session.last_partial_text
        session.metadata[_CARRIED_AUDIO_START_NS] = carried_audio_start_ns
        session.metadata[_CARRIED_MODEL_INFO] = dict(carried_model_info or {})


def merge_carried_partial(session: ASRSession, text: str) -> str:
    """Prepend text retained after a split commit without duplicating it."""
    carried = str(session.metadata.get(_CARRIED_PARTIAL_TEXT, "")).strip()
    current = text.strip()
    if not carried:
        return current
    if not current:
        return carried
    carried_key = " ".join(carried.split()).casefold()
    current_key = " ".join(current.split()).casefold()
    if current_key == carried_key or current_key.startswith(f"{carried_key} "):
        merged = current
    else:
        merged = f"{carried} {current}"
    session.last_partial_text = merged
    return merged


def carried_partial_context(session: ASRSession) -> tuple[str, int | None, dict]:
    return (
        str(session.metadata.get(_CARRIED_PARTIAL_TEXT, "")).strip(),
        session.metadata.get(_CARRIED_AUDIO_START_NS),
        dict(session.metadata.get(_CARRIED_MODEL_INFO, {})),
    )


def next_pcm_chunk(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None
