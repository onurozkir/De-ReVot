"""Explicit native HUD/hook smoke check. Never runs in the ordinary test suite."""

import ctypes as ct
from ctypes import wintypes as wt
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_translator.config.models import OverlayConfig
from voice_translator.desktop.hotkeys import GlobalHotkeys
from voice_translator.desktop.overlay import DesktopOverlay


def main():
    if sys.platform != "win32":
        raise RuntimeError("Run this smoke check on native Windows.")
    user32 = ct.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wt.HWND
    user32.GetWindowLongPtrW.argtypes = [wt.HWND, ct.c_int]
    user32.GetWindowLongPtrW.restype = ct.c_ssize_t
    user32.SendMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    user32.SendMessageW.restype = ct.c_ssize_t
    foreground = user32.GetForegroundWindow()
    overlay = DesktopOverlay(OverlayConfig())
    hooks = GlobalHotkeys({"ptt": "V", "mode": "F9", "pause": "Ctrl+Shift+T", "mute": "Ctrl+Shift+M"})
    try:
        overlay.start()
        hooks.start()
        overlay.state.update({"type": "incoming_committed", "sequence_id": 0,
                              "translated_text": "Canlı altyazı testi — Teşekkür ederim."})
        time.sleep(0.2)
        style = user32.GetWindowLongPtrW(overlay.hwnd, -20)
        hit_test = user32.SendMessageW(overlay.hwnd, 0x84, 0, 0)
        assert style & DesktopOverlay.EX_STYLE == DesktopOverlay.EX_STYLE
        assert hit_test == -1
        assert hooks.running and overlay.running
        assert user32.GetForegroundWindow() == foreground, "HUD must not steal focus"
        print(json.dumps({"status": "MEASURED", "overlay_created": True, "ex_style": hex(style),
                          "hit_test": hit_test, "focus_preserved": True, "hooks_installed": True}))
    finally:
        hooks.stop()
        overlay.stop()
    assert not hooks.running and not overlay.running
    print(json.dumps({"native_workers_stopped": True}))


if __name__ == "__main__":
    main()
