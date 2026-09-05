"""Abstract Base Class for Machine Translation Adapters."""

from __future__ import annotations

import abc
import time
from typing import Optional

from voice_translator.core.types import TranslationEvent, UtteranceEvent


class MTAdapter(abc.ABC):
    """Contract for translation backends."""

    @abc.abstractmethod
    def initialize(
        self,
        *,
        pair_paths: Optional[dict[str, str]] = None,
        nllb_model_path: Optional[str] = None,
        languages: Optional[dict[str, object]] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """Initialize installed offline models from the language registry."""
        pass

    @abc.abstractmethod
    def warmup(self):
        """Warm up TR->EN and EN->TR translation models."""
        pass

    @abc.abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool = False,
        context: Optional[str] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> str:
        """Translate text string from source_lang to target_lang."""
        pass

    def translate_event(
        self,
        event: UtteranceEvent,
        target_lang: str,
        context: Optional[str] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> TranslationEvent:
        """Helper to translate an UtteranceEvent directly into a TranslationEvent."""
        translated = self.translate(
            text=event.text,
            source_lang=event.source_language,
            target_lang=target_lang,
            is_partial=not event.is_final,
            context=context,
            glossary=glossary,
        )
        return TranslationEvent(
            meeting_id=event.meeting_id,
            utterance_id=event.utterance_id,
            sequence_id=event.sequence_id,
            revision=event.revision,
            direction=event.direction,
            source_language=event.source_language,
            target_language=target_lang,
            source_text=event.text,
            translated_text=translated,
            state=event.state,
            created_at_ns=time.monotonic_ns(),
            model_info=event.model_info,
        )

    @abc.abstractmethod
    def shutdown(self):
        """Release resources."""
        pass

