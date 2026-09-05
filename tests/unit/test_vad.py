import numpy as np

from voice_translator.streaming.vad import SileroVAD


def make_energy_vad() -> SileroVAD:
    return SileroVAD(
        threshold=0.6, end_threshold=0.3,
        min_speech_duration_ms=40, min_silence_duration_ms=20,
        start_confirm_frames=2, end_confirm_frames=2, hangover_ms=0,
        load_model=False,
    )


def test_silence_and_low_energy_never_activate_vad():
    vad = make_energy_vad()
    for _ in range(100):
        result = vad.process(np.zeros(320, dtype=np.float32))
        assert not result.active
        assert result.phase == "idle"
    for _ in range(20):
        result = vad.process(np.full(320, 0.002, dtype=np.float32))
        assert not result.active


def test_vad_hysteresis_requires_start_and_end_confirmation():
    vad = make_energy_vad()
    loud = np.full(320, 0.1, dtype=np.float32)
    quiet = np.zeros(320, dtype=np.float32)
    first = vad.process(loud)
    second = vad.process(loud)
    assert not first.active and first.phase == "start_confirm"
    assert second.active and second.transition == "started"
    ending = vad.process(quiet)
    ended = vad.process(quiet)
    assert ending.active and ending.phase == "end_confirm"
    assert not ended.active and ended.transition == "ended"
    assert ended.voiced_ms >= 40
    assert 0 < ended.voiced_ratio < 1

