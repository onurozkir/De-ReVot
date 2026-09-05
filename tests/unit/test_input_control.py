import numpy as np

from voice_translator.streaming.input_control import InputGate


def test_timestamped_press_release_preserve_preroll_and_exact_last_sample():
    gate = InputGate(1000, mode="ptt", preroll_ms=150)
    gate.change("ptt", True, 650_000_000)
    gate.change("ptt", False, 800_000_000)
    actions = gate.route(np.arange(1000, dtype=np.float32), 1_000_000_000)
    admitted = np.concatenate([a.audio for a in actions if a.kind == "audio"])
    np.testing.assert_array_equal(admitted, np.arange(500, 800))
    assert [a.at_ns for a in actions if a.kind == "ptt_release"] == [800_000_000]
    assert len(gate._preroll) == 150
    assert not gate.pressed


def test_closed_gate_never_admits_audio_and_preroll_stays_bounded():
    gate = InputGate(16000, mode="ptt")
    for i in range(1000):
        assert gate.route(np.ones(320, dtype=np.float32), (i + 1) * 20_000_000) == []
    assert len(gate._preroll) == 2400


def test_rapid_taps_and_auto_repeat_emit_each_release_once():
    gate = InputGate(1000, mode="ptt", preroll_ms=0)
    for value, at in [(True, 100), (True, 110), (False, 200), (False, 210), (True, 300), (False, 400)]:
        gate.change("ptt", value, at * 1_000_000)
    actions = gate.route(np.ones(500), 500_000_000)
    assert [a.at_ns for a in actions if a.kind == "ptt_release"] == [200_000_000, 400_000_000]
    assert sum(len(a.audio) for a in actions if a.kind == "audio") == 200


def test_mode_switch_clears_uncommitted_input_and_ptt_hold():
    gate = InputGate(1000, mode="vad")
    gate.change("mode", "ptt", 250_000_000)
    actions = gate.route(np.ones(500), 500_000_000)
    assert [a.kind for a in actions] == ["audio", "reset"]
    assert len(actions[0].audio) == 250
    assert not gate.pressed


def test_overload_fails_closed_instead_of_losing_key_release():
    gate = InputGate(1000, mode="ptt")
    for i in range(100):
        gate.change("ptt", bool(i % 2), i)
    actions = gate.route(np.ones(1000), 1_000_000_000)
    assert [a.kind for a in actions] == ["control_overload"]
    assert gate.muted and not gate.pressed


def test_release_without_new_audio_still_flushes_immediately():
    gate = InputGate(1000, mode="ptt")
    gate.change("ptt", True, 0)
    gate.route(np.ones(100), 100_000_000)
    gate.change("ptt", False, 100_000_001)
    assert [a.kind for a in gate.route(np.empty(0), 100_000_001)] == ["ptt_release"]
