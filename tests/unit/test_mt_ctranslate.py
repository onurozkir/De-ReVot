"""Unit tests for CTranslate2 MT Adapter with INT8, glossary, and context priming."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from voice_translator.core.types import Direction, UtteranceEvent, UtteranceState
from voice_translator.translation.ctranslate_backend import (
    CTranslate2MTAdapter,
    _apply_glossary,
    _resolve_model_dir,
)


def test_apply_glossary_boundary_matching():
    glossary = {
        "pull request": "pull request",
        "deploy": "deployment",
        "pipeline": "pipeline",
    }
    # Should replace exact matching word boundaries
    text = "Yeni bir pull request açtım ve deploy ettik."
    out = _apply_glossary(text, glossary)
    assert "pull request" in out
    assert "deployment" in out

    # Should not mangle substrings inside other words
    glossary_sub = {"art": "skill"}
    text_sub = "This article is smart."
    out_sub = _apply_glossary(text_sub, glossary_sub)
    assert out_sub == "This article is smart."


def test_resolve_model_dir_prefers_ct2_when_available(tmp_path):
    base_dir = tmp_path / "opus-mt-tr-en"
    base_dir.mkdir()
    (base_dir / "pytorch_model.bin").write_text("dummy")

    ct2_dir = tmp_path / "opus-mt-tr-en-ct2"
    ct2_dir.mkdir()
    (ct2_dir / "model.bin").write_text("dummy")

    resolved = _resolve_model_dir(str(base_dir))
    assert resolved == ct2_dir


def test_ctranslate_translates_with_glossary_and_eos():
    tokens_translated = []

    class MockTokenizer:
        eos_token = "</s>"

        def tokenize(self, text):
            return text.split()

        def convert_tokens_to_ids(self, tokens):
            return list(range(len(tokens)))

        def decode(self, token_ids, skip_special_tokens=True):
            return "Sound delay must stay below one second."

    class MockTranslator:
        def translate_batch(self, batch, beam_size=1):
            tokens_translated.extend(batch[0])
            return [SimpleNamespace(hypotheses=[["Sound", "delay", "must", "stay", "below", "one", "second."]])]

    adapter = CTranslate2MTAdapter(beam_size=2)
    adapter.backend_type = "ctranslate2"
    adapter.model_family = "opus"
    adapter.tr_en_translator = MockTranslator()
    adapter.tr_en_tokenizer = MockTokenizer()

    glossary = {"Sound delay": "Audio latency"}
    res = adapter.translate(
        "Ses gecikmesi bir saniyenin altında kalmalı.",
        source_lang="tr",
        target_lang="en",
        glossary=glossary,
    )
    # Check that EOS was appended before sending to CTranslate2
    assert tokens_translated[-1] == "</s>"
    # Check that glossary was applied to result
    assert res == "Audio latency must stay below one second."


def test_translate_event_helper():
    adapter = CTranslate2MTAdapter(beam_size=2)
    adapter.translate = MagicMock(return_value="Hello world")

    event = UtteranceEvent(
        meeting_id="m1",
        stream_id="s1",
        direction=Direction.OUTGOING,
        utterance_id="u1",
        sequence_id=1,
        revision=1,
        state=UtteranceState.COMMITTED,
        source_language="tr",
        text="Merhaba dünya",
        audio_start_ns=100,
        audio_end_ns=200,
        is_final=True,
    )

    t_event = adapter.translate_event(event, "en", context="Önceki cümle.", glossary={"test": "test"})
    assert t_event.translated_text == "Hello world"
    assert t_event.source_text == "Merhaba dünya"
    assert t_event.sequence_id == 1
    adapter.translate.assert_called_once_with(
        text="Merhaba dünya",
        source_lang="tr",
        target_lang="en",
        is_partial=False,
        context="Önceki cümle.",
        glossary={"test": "test"},
    )

