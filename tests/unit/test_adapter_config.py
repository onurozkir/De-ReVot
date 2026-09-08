import asyncio
from types import SimpleNamespace

from voice_translator.config.loader import load_config
from voice_translator.core.types import MeetingStatus
from voice_translator.streaming import orchestrator as orchestrator_module
from voice_translator.translation.ctranslate_backend import CTranslate2MTAdapter
from voice_translator.tts.base import VoiceProfile


def test_ctranslate_uses_configured_beam_for_commits_and_greedy_partials():
    beams = []

    class FakeTokenizer:
        def encode(self, text):
            return [1]

        def convert_ids_to_tokens(self, token_ids):
            return ["token"]

        def convert_tokens_to_ids(self, tokens):
            return [1]

        def decode(self, token_ids):
            return "translated"

    class FakeTranslator:
        def translate_batch(self, tokens, beam_size):
            beams.append(beam_size)
            return [SimpleNamespace(hypotheses=[["token"]])]

    adapter = CTranslate2MTAdapter(beam_size=4)
    adapter.backend_type = "ctranslate2"
    adapter.opus_pairs[("tr", "en")] = (FakeTranslator(), FakeTokenizer())
    adapter.opus_backends[("tr", "en")] = "ctranslate2"

    assert adapter.translate("Merhaba", "tr", "en", is_partial=True) == "translated"
    assert adapter.translate("Merhaba", "tr", "en", is_partial=False) == "translated"
    assert beams == [1, 4]


def test_orchestrator_wires_runtime_config_into_adapters(monkeypatch, tmp_path):
    received = {}

    class FakeAdapter:
        def initialize(self, **kwargs):
            pass

        def warmup(self):
            pass

        def prepare_voice_profile(self, profile):
            pass

    class FakeASR(FakeAdapter):
        def __init__(self, **kwargs):
            received["asr"] = kwargs

    class FakeMT(FakeAdapter):
        def __init__(self, **kwargs):
            received["mt"] = kwargs

    class FakeTTS(FakeAdapter):
        def __init__(self, **kwargs):
            received["tts"] = kwargs

    monkeypatch.setattr(orchestrator_module, "WhisperASRAdapter", FakeASR)
    monkeypatch.setattr(orchestrator_module, "CTranslate2MTAdapter", FakeMT)
    monkeypatch.setattr(orchestrator_module, "XTTSv2Adapter", FakeTTS)

    config = load_config()
    config.voice.profiles_root = str(tmp_path / "voices")
    config.asr.beam_size = 3
    config.translation.beam_size = 4
    config.tts.temperature = 0.6
    config.tts.speed = 1.1
    instance = orchestrator_module.MeetingOrchestrator(config, use_mocks=False)
    instance.profile_manager.profiles["test"] = VoiceProfile(
        "test", "Test", "xtts_v2", str(tmp_path / "reference.wav"), is_default=True
    )

    async def no_op():
        pass

    monkeypatch.setattr(instance.persistence, "start", no_op)
    asyncio.run(instance.initialize_and_warmup())

    assert instance.status == MeetingStatus.READY
    assert received["asr"]["partial_interval_ms"] == 500
    assert received["asr"]["min_audio_rms"] == 0.003
    assert received["asr"]["beam_size"] == 3
    assert callable(received["asr"]["on_inference_wait"])
    assert received["tts"] == {
        "temperature": 0.6,
        "speed": 1.1,
        "top_p": 0.85,
        "repetition_penalty": 2.0,
        "peak_normalization": True,
        "stream_chunk_size": 8,
    }
