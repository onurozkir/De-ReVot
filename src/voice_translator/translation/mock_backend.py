"""Deterministic mock MT adapter for tests."""

from __future__ import annotations

from voice_translator.translation.base import MTAdapter


class MockMTAdapter(MTAdapter):
    """Mock translation adapter for tests."""

    def __init__(self):
        self.is_initialized = False
        self.is_warmed = False

    def initialize(
        self,
        *,
        pair_paths: Optional[dict[str, str]] = None,
        nllb_model_path: Optional[str] = None,
        languages: Optional[dict[str, object]] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        **kwargs,
    ):
        self.is_initialized = True

    def warmup(self):
        self.is_warmed = True

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        is_partial: bool = False,
        context: Optional[str] = None,
        glossary: Optional[dict[str, str]] = None,
    ) -> str:
        if not text:
            return ""
        if glossary:
            for k, v in glossary.items():
                text = text.replace(k, v)
        if source_lang.lower().startswith("tr"):
            if target_lang.lower().startswith("fr"):
                return f"[FR] {text}"
            return f"[EN] {text}"
        return f"[TR] {text}"

    def shutdown(self):
        self.is_initialized = False
        self.is_warmed = False

