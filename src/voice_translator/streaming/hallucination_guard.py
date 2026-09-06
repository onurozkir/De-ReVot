"""Speech-evidence gate for preventing silence/noise hallucinations from committing."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DEFAULT_PATTERNS = (
    "izlediğiniz için teşekkürler", "izlediğiniz için teşekkür ederim",
    "izlediğiniz için çok teşekkür ederim", "izlediğiniz için çok teşekkürler",
    "bizi izlediğiniz için teşekkürler", "bizi izlediğiniz için teşekkür ederiz",
    "abone olmayı unutmayın", "beğenmeyi unutmayın", "görüşmek üzere",
    "abone ol", "thank you for watching", "thanks for watching", "please subscribe", "altyazı m.k.",
    "altyazı m.k", "altyazı", "see you next time", "[music]", "[müzik]", "[applause]", "[alkış]",
    "(laughter)", "(gülüşmeler)", "subtitle", "subtitles",
)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    # Turkish capital İ case-folds to "i" + COMBINING DOT ABOVE. Collapse that
    # search-equivalent form so known hallucinations match regardless of casing.
    normalized = normalized.replace("i\u0307", "i")
    normalized = re.sub(r"[\s\u00a0]+", " ", normalized)
    return normalized.strip(" .,!?:;\"'")


@dataclass(slots=True)
class SpeechEvidence:
    utterance_ms: float
    voiced_ms: float
    voiced_ratio: float
    max_queue_age_ms: float = 0.0


@dataclass(slots=True)
class HallucinationPolicy:
    min_utterance_ms: float = 280.0
    min_voiced_ms: float = 220.0
    min_voiced_ratio: float = 0.25
    max_audio_queue_age_ms: float = 600.0
    max_no_speech_prob: float = 0.80
    min_avg_logprob: float = -1.20
    max_compression_ratio: float = 2.40
    # Conservative, configurable plausibility ceiling, not a linguistic guarantee.
    max_chars_per_second: float = 50.0
    patterns: Sequence[str] = field(default_factory=lambda: DEFAULT_PATTERNS)


@dataclass(slots=True)
class GuardDecision:
    accepted: bool
    reason: str
    normalized_text: str


class HallucinationGuard:
    """Acoustic evidence and confidence primary; known hallucination patterns as final safety net."""

    def __init__(self, policy: HallucinationPolicy | None = None):
        self.policy = policy or HallucinationPolicy()
        self._patterns = {normalize_text(p) for p in self.policy.patterns}

    def evaluate(self, text: str, evidence: SpeechEvidence, model_info: Mapping[str, Any] | None = None) -> GuardDecision:
        normalized = normalize_text(text)
        if not normalized:
            return GuardDecision(False, "empty_text", normalized)
        values = (evidence.utterance_ms, evidence.voiced_ms, evidence.voiced_ratio, evidence.max_queue_age_ms)
        if any(not math.isfinite(value) or value < 0 for value in values):
            return GuardDecision(False, "invalid_speech_evidence", normalized)
        if evidence.max_queue_age_ms > self.policy.max_audio_queue_age_ms:
            return GuardDecision(False, "stale_audio", normalized)
        if evidence.utterance_ms < self.policy.min_utterance_ms:
            return GuardDecision(False, "insufficient_utterance", normalized)
        if evidence.voiced_ms < self.policy.min_voiced_ms:
            return GuardDecision(False, "insufficient_voiced_audio", normalized)
        if evidence.voiced_ratio < self.policy.min_voiced_ratio:
            return GuardDecision(False, "low_voiced_ratio", normalized)
        char_count = sum(char.isalnum() for char in normalized)
        if not char_count:
            return GuardDecision(False, "empty_text", normalized)
        if char_count * 1000.0 > evidence.voiced_ms * self.policy.max_chars_per_second:
            return GuardDecision(False, "implausible_speech_rate", normalized)

        info = model_info or {}
        for key in ("no_speech_prob", "avg_logprob", "compression_ratio"):
            if info.get(key) is not None:
                try:
                    valid = math.isfinite(float(info[key]))
                except (ValueError, TypeError):
                    valid = False
                if not valid:
                    return GuardDecision(False, "invalid_model_evidence", normalized)
        no_speech_prob = info.get("no_speech_prob")
        if no_speech_prob is not None and float(no_speech_prob) > self.policy.max_no_speech_prob:
            return GuardDecision(False, "whisper_no_speech", normalized)
        avg_logprob = info.get("avg_logprob")
        if avg_logprob is not None and float(avg_logprob) < self.policy.min_avg_logprob:
            return GuardDecision(False, "whisper_low_logprob", normalized)
        compression_ratio = info.get("compression_ratio")
        if compression_ratio is not None and float(compression_ratio) > self.policy.max_compression_ratio:
            return GuardDecision(False, "whisper_repetition", normalized)

        # Final safety net: check against known hallucination patterns
        if normalized in self._patterns or normalized.strip("[]()") in self._patterns:
            return GuardDecision(False, "known_hallucination_pattern", normalized)

        words = normalized.split()
        if len(words) >= 8 and len(set(words)) / len(words) < 0.35:
            return GuardDecision(False, "repetitive_text", normalized)
        return GuardDecision(True, "accepted", normalized)

    def is_known_pattern(self, text: str) -> bool:
        normalized = normalize_text(text)
        return normalized in self._patterns or normalized.strip("[]()") in self._patterns

