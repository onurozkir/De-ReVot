import numpy as np
import pytest
from voice_translator.asr.mock_backend import MockASRAdapter
from voice_translator.core.types import Direction
from voice_translator.translation.mock_backend import MockMTAdapter
from voice_translator.tts.base import VoiceProfile
from voice_translator.tts.mock_backend import MockTTSAdapter


def test_mock_asr_adapter():
    asr = MockASRAdapter(predefined_transcripts=["Test cümle"])
    asr.initialize()
    asr.warmup()
    assert asr.is_warmed

    session = asr.create_session("stream_1", Direction.OUTGOING, "tr")
    chunk = np.zeros(8000, dtype=np.float32)
    ev = asr.process_audio(session, chunk, captured_at_ns=1000)
    assert ev is not None
    assert ev.text == "Test cümle"

    commit_ev = asr.flush_session(session)
    assert commit_ev is not None
    assert commit_ev.is_final


def test_mock_mt_adapter():
    mt = MockMTAdapter()
    mt.initialize()
    mt.warmup()

    res_en = mt.translate("Merhaba", "tr", "en")
    assert "[EN] Merhaba" == res_en

    res_fr = mt.translate("Merhaba", "tr", "fr")
    assert "[FR] Merhaba" == res_fr

    res_tr = mt.translate("Hello", "en", "tr")
    assert "[TR] Hello" == res_tr


def test_mock_tts_adapter():
    tts = MockTTSAdapter(sample_rate=24000)
    tts.initialize()
    tts.warmup()

    prof = VoiceProfile(
        id="test_voice",
        display_name="Test Voice",
        backend="mock",
        reference_audio_path="test.wav",
    )
    chunks = list(tts.synthesize_committed("Hello world", prof, "en"))
    assert len(chunks) > 0
    assert chunks[0].dtype == np.float32

