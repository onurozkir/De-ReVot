"""Commit Controller for stable-prefix and semantic boundary commits."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

from voice_translator.core.types import UtteranceEvent, UtteranceState


@dataclass
class CommitDecision:
    should_commit: bool
    committed_text: str
    remaining_partial_text: str
    reason: str  # "punctuation", "stable_prefix", "silence_endpoint", "deadline_timeout"


# Punctuation / conjunction clause delimiters
_PUNCT_REGEX = re.compile(r'([.?!;:]+)(\s+|$)', re.UNICODE)
_CONJUNCTION_REGEX = re.compile(r'(\s+(?:ve|veya|ama|fakat|ancak|çünkü|ise|ki|böylece)\s+)', re.IGNORECASE | re.UNICODE)

# Turkish SOV (Subject-Object-Verb) predicate suffix patterns
_TURKISH_VERB_SUFFIXES = (
    # Past: -di/-ti family with personal markers
    r'[a-zçğıöşü]+(?:[dD][ıİuÜ]|[tT][ıİuÜ])(?:[kKmMnN]|n[ıİzZ]|l[aA]r)?'
    # Present continuous: -iyor family
    r'|[a-zçğıöşü]+[ıİuÜ]yor(?:[uU]m|[uU]z|[sS][uU]n|[sS][uU]n[uU]z|l[aA]r)?'
    # Future: -ecek/-acak family
    r'|[a-zçğıöşü]+(?:[eE]ce|[aA]ca)[gğkK](?:[ıİuÜ]m|[ıİuÜ]z|[sS][ıİ][nN]|l[aA]r)?'
    # Aorist: -ir/-er/-ar family
    r'|[a-zçğıöşü]+(?:[ıİuÜeEaA]r)(?:[ıİuÜeEaA]m|[ıİuÜeEaA]z|[sS][ıİ][nN])?'
    # Evidential: -miş family
    r'|[a-zçğıöşü]+(?:[mM][ıİuÜ]ş)(?:[tT][ıİ]r|[ıİuÜ]m|[ıİuÜ]z|[sS][ıİ][nN])?'
    # Necessitative/Optative/Imperative: -meli/-elim family
    r'|[a-zçğıöşü]+(?:[mM][eE]li|[mM][aA]l[ıİ]|[eE]lim|[aA]l[ıİ]m|[sS][iİ]n|[sS][ıİ]n|[sS][uU]n|[sS][üÜ]n)'
    # Common copulas, auxiliary predicates, and adjectives
    r'|(?:yaptım|yaptık|yaptı|yaptınız|ettim|ettik|etti|oldu|bitti|geldi|gittim|gitti|'
    r'gördüm|gördük|aldım|aldık|verdim|verdik|başladı|bitirdi|vardı|yoktu|değil|'
    r'değilim|değiliz|var|yok|tamam|hazır|doğru|iyi|güzel|mümkün|lazım|gerek|olur)'
    # Question particles
    r'|(?:mı|mi|mu|mü|mıyım|miyiz|musun|müsünüz)'
)
_TURKISH_PREDICATE_REGEX = re.compile(r'^(?:' + _TURKISH_VERB_SUFFIXES + r')$', re.IGNORECASE | re.UNICODE)

_TURKISH_OPEN_CONJUNCTIONS = {
    "ve", "veya", "ama", "fakat", "ancak", "çünkü", "ile", "ise", "ki",
    "lakin", "halbuki", "oysa", "veyahut", "gibi", "için", "diye", "kadar"
}

_TURKISH_NON_PREDICATES = {
    # Non-predicate nouns ending in -tı / -ti / -tu / -tü (deverbal noun suffixes)
    "toplantı", "toplantısı", "toplantılar", "toplantıları",
    "görüntü", "görüntüsü", "görüntüler",
    "alıntı", "alıntılar", "belirti", "belirtiler",
    "yaşantı", "akıntı", "bağıntı", "sıkıntı", "sıkıntılar", "sızıntı", "kasıntı",
    # Common words matching -r aorist pattern but are nouns/determiners
    "bir", "her", "kadar", "tekrar", "karar", "zarar", "fikir", "şehir", "demir",
    "müdür", "doktor", "rapor", "lider", "haber", "biber", "duvar", "pazar",
    # Common nouns/adverbs matching other suffix patterns
    "şirket", "şirkette", "şirketteki", "toplantıdaki",
    "zaman", "insan", "bölüm", "durum", "konu", "taraf", "tarafından",
    "şey", "şeyler", "gün", "hafta", "ay", "yıl", "dakika", "saniye",
}


def is_turkish_predicate_tail(text: str) -> bool:
    """Check whether the final word of text is a complete Turkish finite predicate."""
    words = text.strip().split()
    if not words:
        return False
    tail = words[-1].rstrip(".,?!:;").casefold().replace("i\u0307", "i")
    if not tail or tail in _TURKISH_OPEN_CONJUNCTIONS or tail in _TURKISH_NON_PREDICATES:
        return False
    return bool(_TURKISH_PREDICATE_REGEX.match(tail))


def is_open_conjunction_tail(text: str) -> bool:
    """Check whether the text ends with an open conjunction indicating mid-clause pause."""
    words = text.strip().split()
    if not words:
        return False
    tail = words[-1].rstrip(".,?!:;").casefold().replace("i\u0307", "i")
    return tail in _TURKISH_OPEN_CONJUNCTIONS


class CommitController:
    """Manages irreversible commit decisions from streaming ASR hypotheses."""

    def __init__(
        self,
        min_words: int = 3,
        max_wait_ms: int = 1800,
        stable_prefix_min_count: int = 2,
        enable_adaptive_sov: bool = True,
        sov_min_silence_ms: int = 200,
    ):
        self.min_words = min_words
        self.max_wait_ms = max_wait_ms
        self.stable_prefix_min_count = stable_prefix_min_count
        self.enable_adaptive_sov = enable_adaptive_sov
        self.sov_min_silence_ms = sov_min_silence_ms

        self.last_hypotheses: List[str] = []
        self.first_hypothesis_time_ms: float = 0
        self.stable_prefix_candidate = ""
        self.stable_prefix_matches = 0

        # Punctuation / conjunction clause delimiters
        self._punct_regex = _PUNCT_REGEX
        self._conjunction_regex = _CONJUNCTION_REGEX

    def reset(self):
        self.last_hypotheses.clear()
        self.first_hypothesis_time_ms = 0
        self.stable_prefix_candidate = ""
        self.stable_prefix_matches = 0

    def evaluate(
        self,
        current_text: str,
        is_silence_endpoint: bool = False,
        now_ms: Optional[float] = None,
        silence_ms: float = 0.0,
        language: str = "tr",
    ) -> CommitDecision:
        current_text = current_text.strip()
        if not current_text:
            return CommitDecision(False, "", "", "empty")

        now = now_ms or (time.monotonic() * 1000.0)
        if self.first_hypothesis_time_ms == 0:
            self.first_hypothesis_time_ms = now

        words = current_text.split()
        elapsed_wait_ms = now - self.first_hypothesis_time_ms
        is_tr = language.lower().startswith("tr")

        # 1. Silence Endpoint flush
        if is_silence_endpoint:
            self.reset()
            return CommitDecision(
                should_commit=True,
                committed_text=current_text,
                remaining_partial_text="",
                reason="silence_endpoint",
            )

        # 2. Punctuation boundary check (e.g. "Merhaba nasılsınız? Bugün...")
        punct_match = self._punct_regex.search(current_text)
        if punct_match:
            split_idx = punct_match.end()
            committed = current_text[:split_idx].strip()
            remainder = current_text[split_idx:].strip()
            if len(committed.split()) >= self.min_words or is_silence_endpoint:
                self.reset()
                return CommitDecision(
                    should_commit=True,
                    committed_text=committed,
                    remaining_partial_text=remainder,
                    reason="punctuation",
                )

        # 3. Adaptive Turkish SOV Verb Endpointing
        # When a complete predicate/verb is detected at sentence tail, commit after short silence (e.g. 200ms)
        # instead of waiting out the full ~950ms VAD hangover!
        if self.enable_adaptive_sov and is_tr and len(words) >= self.min_words:
            if is_turkish_predicate_tail(current_text) and silence_ms >= self.sov_min_silence_ms:
                self.reset()
                return CommitDecision(
                    should_commit=True,
                    committed_text=current_text,
                    remaining_partial_text="",
                    reason="turkish_sov_verb",
                )

        # 4. Stable prefix detection across consecutive revisions
        self.last_hypotheses.append(current_text)
        if len(self.last_hypotheses) > 5:
            self.last_hypotheses.pop(0)

        if len(self.last_hypotheses) >= 2:
            common_prefix = self._longest_common_word_prefix(self.last_hypotheses[-2], self.last_hypotheses[-1])
            prefix_words = common_prefix.split()
            if len(prefix_words) >= self.min_words:
                if self._comparison_key(common_prefix) == self._comparison_key(self.stable_prefix_candidate):
                    self.stable_prefix_matches += 1
                else:
                    self.stable_prefix_candidate = common_prefix
                    self.stable_prefix_matches = 1

                if self.stable_prefix_matches >= self.stable_prefix_min_count:
                    # If entire stable utterance ends with a confirmed Turkish predicate, commit early
                    if self.enable_adaptive_sov and is_tr and is_turkish_predicate_tail(common_prefix):
                        self.reset()
                        return CommitDecision(
                            should_commit=True,
                            committed_text=common_prefix,
                            remaining_partial_text=current_text[len(common_prefix):].strip(),
                            reason="turkish_sov_verb",
                        )

                    # Check for conjunction boundary within stable prefix
                    conj_match = self._conjunction_regex.search(common_prefix)
                    if conj_match:
                        split_idx = conj_match.start()
                        committed = common_prefix[:split_idx].strip()
                        remainder = current_text[len(committed):].strip()
                        if len(committed.split()) >= self.min_words:
                            self.reset()
                            return CommitDecision(
                                should_commit=True,
                                committed_text=committed,
                                remaining_partial_text=remainder,
                                reason="stable_prefix_clause",
                            )
            else:
                self.stable_prefix_candidate = ""
                self.stable_prefix_matches = 0

        # 5. Max wait deadline timeout
        effective_max_wait = self.max_wait_ms * 1.5 if (is_tr and is_open_conjunction_tail(current_text)) else self.max_wait_ms
        if elapsed_wait_ms >= effective_max_wait and len(words) >= self.min_words:
            committed, remainder = self._deadline_safe_split(current_text)
            if committed:
                self.reset()
                return CommitDecision(
                    should_commit=True,
                    committed_text=committed,
                    remaining_partial_text=remainder,
                    reason="deadline_timeout",
                )

        return CommitDecision(
            should_commit=False,
            committed_text="",
            remaining_partial_text=current_text,
            reason="buffering_partial",
        )

    def _longest_common_word_prefix(self, s1: str, s2: str) -> str:
        w1 = s1.split()
        w2 = s2.split()
        common = []
        for a, b in zip(w1, w2):
            if self._comparison_key(a) == self._comparison_key(b):
                common.append(a)
            else:
                break
        return " ".join(common)

    @staticmethod
    def _comparison_key(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", " ".join(text.split())).casefold()
        return normalized.replace("i\u0307", "i")

    def _deadline_safe_split(self, current_text: str) -> Tuple[str, str]:
        """Return only text supported by stable revisions or a clause boundary."""
        stable_words = self.stable_prefix_candidate.split()
        current_words = current_text.split()
        if (
            len(stable_words) >= self.min_words
            and len(current_words) >= len(stable_words)
            and all(self._comparison_key(a) == self._comparison_key(b) for a, b in zip(stable_words, current_words))
        ):
            split_at = len(stable_words)
            return " ".join(current_words[:split_at]), " ".join(current_words[split_at:])

        boundaries = [match.end() for match in self._punct_regex.finditer(current_text)]
        boundaries.extend(match.start() for match in self._conjunction_regex.finditer(current_text))
        for split_idx in sorted(boundaries, reverse=True):
            committed = current_text[:split_idx].strip()
            if len(committed.split()) >= self.min_words:
                return committed, current_text[split_idx:].strip()
        return "", current_text
