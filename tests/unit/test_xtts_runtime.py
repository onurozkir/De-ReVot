from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from voice_translator.tts import xtts_backend
from voice_translator.tts.base import VoiceProfile
from voice_translator.tts.conditioning import VoiceProfileManager


def test_coqui_xtts_runtime_imports_are_available():
    assert xtts_backend.torch is not None
    assert xtts_backend.XttsConfig is not None, xtts_backend._tts_import_error
    assert xtts_backend.Xtts is not None, xtts_backend._tts_import_error


def test_xtts_uses_configured_temperature_and_speed(tmp_path):
    received = {}

    class FakeModel:
        def inference(self, **kwargs):
            received.update(kwargs)
            return {"wav": [0.0, 0.1]}

    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"voice")
    profile = VoiceProfile("onur", "Onur", "xtts_v2", str(reference))
    adapter = xtts_backend.XTTSv2Adapter(temperature=0.55, speed=1.15)
    adapter.model = FakeModel()
    cache_key = f"{profile.id}_{VoiceProfileManager.compute_audio_hash(str(reference))}"
    adapter._latents_cache[cache_key] = (SimpleNamespace(), SimpleNamespace())
    adapter._prepared_profiles[(profile.id, tuple(profile.all_reference_paths))] = cache_key

    chunks = list(adapter.synthesize_committed("Hello", profile, "en"))

    assert len(chunks) == 1
    assert chunks[0].dtype == np.float32
    assert received["temperature"] == 0.55
    assert received["speed"] == 1.15


def test_xtts_missing_reference_fails_instead_of_using_zero_latents(tmp_path):
    class FakeModel:
        def inference(self, **kwargs):
            pytest.fail("inference must not run without voice conditioning")

    profile = VoiceProfile(
        "missing",
        "Missing Voice",
        "xtts_v2",
        str(tmp_path / "missing.wav"),
    )
    adapter = xtts_backend.XTTSv2Adapter()
    adapter.model = FakeModel()

    with pytest.raises(RuntimeError, match="Voice reference audio.*not found"):
        list(adapter.synthesize_committed("Hello", profile, "en"))


def test_xtts_soundfile_workaround_is_scoped_to_conditioning_call(tmp_path):
    reference = tmp_path / "reference.wav"
    sf.write(reference, 0.1 * np.sin(2 * np.pi * 440 * np.arange(48000) / 16000), 16000)
    profile = VoiceProfile(
        "onur",
        "Onur",
        "xtts_v2",
        str(reference),
        conditioning_cache_path=str(tmp_path / "cache"),
    )
    original_torchaudio_load = xtts_backend.torchaudio.load
    original_xtts_load_audio = xtts_backend.xtts_model_module.load_audio
    observed = {}

    class FakeModel:
        def get_conditioning_latents(self, **kwargs):
            observed["torchaudio_load"] = xtts_backend.torchaudio.load
            observed["xtts_load_audio"] = xtts_backend.xtts_model_module.load_audio
            return SimpleNamespace(), SimpleNamespace()

    adapter = xtts_backend.XTTSv2Adapter()
    adapter.model = FakeModel()
    adapter.prepare_voice_profile(profile)

    assert original_torchaudio_load is not xtts_backend._soundfile_load
    assert observed["torchaudio_load"] is original_torchaudio_load
    assert observed["xtts_load_audio"] is xtts_backend._soundfile_load
    assert xtts_backend.torchaudio.load is original_torchaudio_load
    assert xtts_backend.xtts_model_module.load_audio is original_xtts_load_audio
