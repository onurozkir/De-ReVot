import pytest

from voice_translator.desktop.hotkeys import GlobalHotkeys, parse_hotkey, validate_bindings
from voice_translator.desktop.overlay import SubtitleState


def test_modifier_chords_release_and_repeat_do_not_toggle_twice():
    hooks = GlobalHotkeys({"ptt": "Mouse 4", "pause": "Ctrl+Shift+T"})
    hooks.key_edge(5, True, 1)
    hooks.key_edge(5, True, 2)
    hooks.key_edge(5, False, 3)
    for key in (162, 160, 84, 84):
        hooks.key_edge(key, True, 4)
    assert hooks.drain() == [("ptt", True, 1), ("ptt", False, 3), ("pause", True, 4)]


@pytest.mark.parametrize("key", ["", "Ctrl", "Ctrl+Ctrl+V", "F99", "V+NoSuchKey"])
def test_invalid_keys_fail_visibly(key):
    with pytest.raises(ValueError):
        parse_hotkey(key)


def test_overlapping_global_controls_rejected():
    with pytest.raises(ValueError, match="overlap"):
        validate_bindings({"ptt": "T", "pause": "Ctrl+Shift+T"})


def test_hotkey_queue_overflow_reports_fault_and_discards_stale_edges():
    hooks = GlobalHotkeys({"ptt": "V"})
    for i in range(100):
        hooks.key_edge(86, i % 2 == 0)
    assert [event[0] for event in hooks.drain()] == ["fault"]


def test_subtitle_revision_commit_order_and_expiry():
    state = SubtitleState(expiry_sec=5)
    state.update({"type": "incoming_partial", "sequence_id": 0, "revision": 2, "translated_text": "yeni"}, now=1)
    state.update({"type": "incoming_partial", "sequence_id": 0, "revision": 1, "translated_text": "eski"}, now=2)
    assert state.text(now=2) == "yeni"
    state.update({"type": "incoming_committed", "sequence_id": 0, "translated_text": "Merhaba"}, now=3)
    state.update({"type": "incoming_partial", "sequence_id": 0, "revision": 99, "translated_text": "geç"}, now=4)
    assert state.text(now=4) == "Merhaba"
    assert state.text(now=9) == ""
    state.clear()
    state.update({"type": "incoming_committed", "sequence_id": 0, "translated_text": "Yeni oturum"}, now=10)
    assert state.text(now=10) == "Yeni oturum"


def test_subtitle_state_is_bounded_during_long_sessions():
    state = SubtitleState()
    for i in range(1000):
        state.update({"type": "incoming_committed", "sequence_id": i, "translated_text": "x" * 1000})
    assert len(state.commits) == 2
    assert len(state.text()) <= 600
