from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import json

from voice_translator.tts import xtts_backend
from voice_translator.tts.base import VoiceProfile
from voice_translator.tts.conditioning import VoiceProfileManager


def test_xtts_hyperparameters_validation():
    # Valid parameters
    adapter = xtts_backend.XTTSv2Adapter(
        temperature=0.65,
        speed=1.0,
        top_p=0.85,
        repetition_penalty=2.0,
        peak_normalization=True,
    )
    assert adapter.temperature == 0.65
    assert adapter.top_p == 0.85
    assert adapter.repetition_penalty == 2.0
    assert adapter.peak_normalization is True

    # Invalid top_p
    with pytest.raises(ValueError, match="top_p"):
        xtts_backend.XTTSv2Adapter(top_p=0.0)
    with pytest.raises(ValueError, match="top_p"):
        xtts_backend.XTTSv2Adapter(top_p=1.5)

    # Invalid repetition_penalty
    with pytest.raises(ValueError, match="repetition_penalty"):
        xtts_backend.XTTSv2Adapter(repetition_penalty=-1.0)


def test_xtts_inference_passes_hyperparameters(tmp_path):
    received = {}

    class FakeModel:
        def inference_stream(self, **kwargs):
            received.update(kwargs)
            # Output audio with peak 1.5 (clipping)
            yield np.array([0.0, 1.5, -1.2, 0.5])

    ref1 = tmp_path / "reference.wav"
    ref1.write_bytes(b"sample1")

    profile = VoiceProfile(
        id="speaker_1",
        display_name="Speaker 1",
        backend="xtts_v2",
        reference_audio_path=str(ref1),
    )

    adapter = xtts_backend.XTTSv2Adapter(
        temperature=0.65,
        speed=1.05,
        top_p=0.82,
        repetition_penalty=2.2,
        peak_normalization=True,
    )
    adapter.model = FakeModel()
    cache_key = f"{profile.id}_{VoiceProfileManager.compute_audio_hash(str(ref1))}"
    adapter._latents_cache[cache_key] = (SimpleNamespace(), SimpleNamespace())
    adapter._prepared_profiles[(profile.id, tuple(profile.all_reference_paths))] = cache_key

    chunks = list(adapter.synthesize_committed("Hello world", profile, "en"))

    assert len(chunks) == 1
    pcm = chunks[0]
    assert received["temperature"] == 0.65
    assert received["speed"] == 1.05
    assert received["top_p"] == 0.82
    assert received["repetition_penalty"] == 2.2

    # Streaming applies a fixed ceiling, never future-dependent gain scaling.
    peak = float(np.max(np.abs(pcm)))
    assert pytest.approx(peak, abs=1e-4) == 0.89125
    assert peak < 1.0
    assert pcm[-1] == .5


def test_multi_sample_hash_and_conditioning(tmp_path):
    ref1 = tmp_path / "reference_1.wav"
    ref2 = tmp_path / "reference_2.wav"
    write_wav(ref1)
    write_wav(ref2, frequency=600)

    profile = VoiceProfile(
        id="multi_speaker",
        display_name="Multi Speaker",
        backend="xtts_v2",
        reference_audio_path=str(ref1),
        reference_audio_paths=[str(ref2)],
        conditioning_cache_path=str(tmp_path / "cache"),
    )

    assert profile.all_reference_paths == [str(ref1.resolve()), str(ref2.resolve())]

    hash_single = VoiceProfileManager.compute_audio_hash(str(ref1))
    hash_multi = VoiceProfileManager.compute_audio_hash(profile.all_reference_paths)

    assert hash_single != ""
    assert hash_multi != ""
    assert hash_single != hash_multi

    observed_audio_paths = []

    class FakeModel:
        def get_conditioning_latents(self, **kwargs):
            observed_audio_paths.extend(kwargs.get("audio_path", []))
            return SimpleNamespace(), SimpleNamespace()

    adapter = xtts_backend.XTTSv2Adapter()
    adapter.model = FakeModel()
    adapter.prepare_voice_profile(profile)

    assert len(observed_audio_paths) == 2
    assert str(ref1.resolve()) in observed_audio_paths
    assert str(ref2.resolve()) in observed_audio_paths


def write_wav(path, seconds=3.0, rate=16000, frequency=440, gain=0.1):
    sf.write(path, gain * np.sin(2 * np.pi * frequency * np.arange(int(seconds * rate)) / rate), rate)


@pytest.mark.parametrize("manifest", [None, {}, {"reference_audio_paths": [f"voice_{i:02}.wav" for i in range(6, 0, -1)]}])
def test_six_reference_discovery_without_phantom_reference(tmp_path, manifest):
    directory = tmp_path / "speaker"
    directory.mkdir()
    for i in range(6, 0, -1):
        write_wav(directory / f"voice_{i:02}.wav")
    (directory / "cache").mkdir()
    write_wav(directory / "cache" / "ignored.wav")
    if manifest is not None:
        (directory / "profile.json").write_text(json.dumps(manifest))
    profile = VoiceProfileManager(str(tmp_path)).get_default_profile()
    assert len(profile.all_reference_paths) == 6
    assert [Path(p).name for p in profile.all_reference_paths] == [f"voice_{i:02}.wav" for i in range(1, 7)]
    assert profile.reference_audio_path == profile.all_reference_paths[0]


def test_legacy_manifest_is_authoritative_and_missing_refs_are_retained(tmp_path):
    directory = tmp_path / "speaker"
    directory.mkdir()
    write_wav(directory / "voice_01.wav")
    (directory / "profile.json").write_text(json.dumps({"reference_audio_path": "missing.wav"}))
    profile = VoiceProfileManager(str(tmp_path)).get_default_profile()
    assert len(profile.all_reference_paths) == 1
    with pytest.raises(RuntimeError, match="missing.wav.*not found"):
        VoiceProfileManager.validate_reference_audio(profile.all_reference_paths)


@pytest.mark.parametrize("kind", ["missing", "tiny", "corrupt", "silent", "nan"])
def test_invalid_reference_fails_with_path(tmp_path, kind):
    path = tmp_path / f"{kind}.wav"
    if kind == "tiny":
        path.write_bytes(b"RIFF")
    elif kind == "corrupt":
        path.write_bytes(b"invalid" * 1000)
    elif kind == "silent":
        write_wav(path, gain=0)
    elif kind == "nan":
        sf.write(path, np.full(16000, np.nan), 16000, subtype="FLOAT")
    with pytest.raises(RuntimeError, match=f"{kind}.wav"):
        VoiceProfileManager.validate_reference_audio([str(path)])


def test_reference_warnings_and_order_independent_hash(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    write_wav(a, seconds=1, gain=1)
    write_wav(b, rate=24000)
    report = VoiceProfileManager.validate_reference_audio([str(a), str(b)])
    assert any("duration" in w for w in report[0]["warnings"])
    assert any("clipped" in w for w in report[0]["warnings"])
    assert all(any("mixed sample rates" in w for w in r["warnings"]) for r in report)
    hash1 = VoiceProfileManager.compute_audio_hash([str(a), str(b)])
    assert hash1 == VoiceProfileManager.compute_audio_hash([str(b), str(a)])
    write_wav(b, frequency=500)
    assert hash1 != VoiceProfileManager.compute_audio_hash([str(a), str(b)])
    b.unlink()
    with pytest.raises(RuntimeError, match="b.wav"):
        VoiceProfileManager.compute_audio_hash([str(a), str(b)])


def test_conditioning_disk_cache_and_no_audio_io_per_utterance(tmp_path, monkeypatch):
    import torch
    paths = [tmp_path / f"voice_{i:02}.wav" for i in range(1, 7)]
    for p in paths:
        write_wav(p)
    calls = []

    class Model:
        def get_conditioning_latents(self, **kwargs):
            calls.append(kwargs)
            return torch.ones(1, 2, 3), torch.ones(1, 4, 1)

        def inference_stream(self, **kwargs):
            yield np.array([0.1], dtype=np.float32)

    profile = VoiceProfile("six", "Six", "xtts_v2", reference_audio_paths=[str(p) for p in reversed(paths)],
                           conditioning_cache_path=str(tmp_path / "cache"))
    adapter = xtts_backend.XTTSv2Adapter()
    adapter.device = "cpu"
    adapter.model = Model()
    adapter.prepare_voice_profile(profile)
    assert calls[0]["audio_path"] == [str(p.resolve()) for p in paths]
    assert calls[0]["gpt_cond_len"] == 30
    second = xtts_backend.XTTSv2Adapter()
    second.device = "cpu"
    second.model = Model()
    second.prepare_voice_profile(profile)
    assert len(calls) == 1
    second._model_identity = "different-checkpoint"
    second.prepare_voice_profile(profile)
    assert len(calls) == 2
    write_wav(paths[-1], frequency=600)
    second.prepare_voice_profile(profile)
    assert len(calls) == 3
    monkeypatch.setattr(VoiceProfileManager, "compute_audio_hash", lambda *args: pytest.fail("audio I/O in synthesis"))
    monkeypatch.setattr(VoiceProfileManager, "validate_reference_audio", lambda *args: pytest.fail("validation in synthesis"))
    assert list(second.synthesize_committed("Hello", profile))
    assert list(second.synthesize_committed("Again", profile))

