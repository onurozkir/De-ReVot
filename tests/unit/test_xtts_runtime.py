from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from voice_translator.tts import xtts_backend
from voice_translator.tts.base import VoiceProfile
from voice_translator.tts.conditioning import VoiceProfileManager


@pytest.mark.parametrize("use_cache", [True, False])
def test_coqui_stream_decodes_only_new_tokens_with_transformers_cache(use_cache):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch = xtts_backend.torch
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=16, n_positions=64, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=0, eos_token_id=None, pad_token_id=0,
    )).eval()
    model.final_norm = torch.nn.Identity()
    xtts_backend._configure_streaming_compat(model)
    lengths = []
    model.register_forward_pre_hook(
        lambda module, args, kwargs: lengths.append(kwargs["input_ids"].shape[-1]),
        with_kwargs=True,
    )
    stream = model.generate_stream(
        torch.tensor([[1, 2, 3]]), max_new_tokens=4, do_sample=True,
        output_hidden_states=True, return_dict_in_generate=True,
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        use_cache=use_cache,
    )
    assert len(list(stream)) == 4
    assert lengths == ([3, 1, 1, 1] if use_cache else [3, 4, 5, 6])


@pytest.mark.parametrize("pcm", [[0.1], [], [float("nan")]])
def test_warmup_bounds_generation_and_requires_valid_pcm(pcm):
    received = {}
    closed = []

    class Model:
        def inference_stream(self, **kwargs):
            received.update(kwargs)
            try:
                yield np.asarray(pcm)
            finally:
                closed.append(True)

    adapter = xtts_backend.XTTSv2Adapter(stream_chunk_size=4)
    adapter.device = "cpu"
    adapter.model = Model()
    adapter._is_warm = True
    if pcm and np.isfinite(pcm).all():
        adapter.warmup()
        assert adapter._is_warm
    else:
        with pytest.raises(xtts_backend.WarmupError, match="PCM"):
            adapter.warmup()
        assert not adapter._is_warm
    assert received["max_new_tokens"] == 8
    assert received["stream_chunk_size"] == 4
    assert closed == [True]


def test_xtts_yields_first_chunk_before_producing_rest_and_closes_on_cancel(tmp_path):
    state = []

    class Model:
        def inference(self, **kwargs):
            pytest.fail("Full-waveform inference cannot provide early PCM")

        def inference_stream(self, **kwargs):
            try:
                state.append("first")
                yield xtts_backend.torch.tensor([[.1, -.2]])
                state.append("second")
                yield xtts_backend.torch.tensor([.3, -.4])
            finally:
                state.append("closed")

    profile = VoiceProfile("test", "Test", "xtts_v2", str(tmp_path / "voice.wav"))
    adapter = xtts_backend.XTTSv2Adapter()
    adapter.model = Model()
    adapter._prepared_profiles[(profile.id, tuple(profile.all_reference_paths))] = "ready"
    adapter._latents_cache["ready"] = (object(), object())
    iterator = adapter.synthesize_committed("First sentence. Next sentence.", profile)
    pcm = next(iterator)
    assert state == ["first"]
    assert pcm.dtype == np.float32 and pcm.ndim == 1
    iterator.close()
    assert state == ["first", "closed"]


def test_coqui_xtts_runtime_imports_are_available():
    assert xtts_backend.torch is not None
    assert xtts_backend.XttsConfig is not None, xtts_backend._tts_import_error
    assert xtts_backend.Xtts is not None, xtts_backend._tts_import_error


def test_xtts_uses_configured_temperature_and_speed(tmp_path):
    received = {}

    class FakeModel:
        def inference_stream(self, **kwargs):
            received.update(kwargs)
            yield np.array([0.0, 0.1])

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
    assert received["stream_chunk_size"] == 8
    assert received["overlap_wav_len"] == 1024
    assert received["enable_text_splitting"] is False


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
