"""Native Windows low-level hooks; callbacks only enqueue bounded control edges."""

import ctypes as ct
from ctypes import wintypes as wt
import queue
import sys
import threading
import time


def parse_hotkey(value: str) -> frozenset[int]:
    names = {"CTRL": 17, "CONTROL": 17, "SHIFT": 16, "ALT": 18,
             "CAPSLOCK": 20, "SPACE": 32, "MOUSE4": 5, "MOUSE5": 6}
    keys = []
    for token in value.upper().replace(" ", "").split("+"):
        if token in names:
            keys.append(names[token])
        elif len(token) == 1 and token.isascii() and token.isalnum():
            keys.append(ord(token))
        elif token.startswith("F") and token[1:].isdigit() and 1 <= int(token[1:]) <= 12:
            keys.append(111 + int(token[1:]))
        else:
            raise ValueError(f"Unsupported hotkey '{value}'. Use V, Caps Lock, Mouse 4/5, F1-F12 or Ctrl/Shift/Alt combinations.")
    if len(set(keys)) != len(keys) or not set(keys) - {16, 17, 18}:
        raise ValueError(f"Hotkey needs a distinct non-modifier key: {value}")
    return frozenset(keys)


def validate_bindings(bindings: dict[str, str]) -> dict[str, frozenset[int]]:
    parsed = {action: parse_hotkey(value) for action, value in bindings.items()}
    values = list(parsed.values())
    if any(a <= b or b <= a for i, a in enumerate(values) for b in values[i + 1:]):
        raise ValueError("Global hotkeys overlap. Choose distinct PTT, mode, pause and mute keys.")
    return parsed


class GlobalHotkeys:
    def __init__(self, bindings: dict[str, str]):
        self.bindings = validate_bindings(bindings)
        self._pressed: set[int] = set()
        self._active = {action: False for action in bindings}
        self._events: queue.Queue = queue.Queue(maxsize=64)
        self._overflow = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self.error: str | None = None

    def key_edge(self, key: int, down: bool, at_ns: int | None = None) -> None:
        # Collapse left/right modifiers, preserving a held sibling modifier.
        if down:
            self._pressed.add(key)
        else:
            self._pressed.discard(key)
        pressed = {({160: 16, 161: 16, 162: 17, 163: 17, 164: 18, 165: 18}.get(k, k)) for k in self._pressed}
        for action, chord in self.bindings.items():
            active = chord <= pressed
            if active != self._active[action]:
                self._active[action] = active
                if action == "ptt" or active:
                    try:
                        self._events.put_nowait((action, active, at_ns if at_ns is not None else time.monotonic_ns()))
                    except queue.Full:
                        self._overflow = True

    def drain(self) -> list[tuple]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        if self._overflow:
            self._overflow = False
            return [("fault", True, time.monotonic_ns())]
        return events

    def start(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Global hotkeys require native Windows.")
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="voice-hotkeys", daemon=True)
        self._thread.start()
        if not self._ready.wait(3) or self.error:
            self.stop()
            raise RuntimeError(self.error or "Windows hotkey initialization timed out.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._pressed.clear()
        self._active = dict.fromkeys(self._active, False)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self.error)

    def _run(self) -> None:
        user32 = ct.WinDLL("user32", use_last_error=True)
        kernel32 = ct.WinDLL("kernel32", use_last_error=True)
        hook_proc = ct.WINFUNCTYPE(ct.c_ssize_t, ct.c_int, wt.WPARAM, wt.LPARAM)
        user32.SetWindowsHookExW.argtypes = [ct.c_int, hook_proc, wt.HINSTANCE, wt.DWORD]
        user32.SetWindowsHookExW.restype = wt.HANDLE
        user32.CallNextHookEx.argtypes = [wt.HANDLE, ct.c_int, wt.WPARAM, wt.LPARAM]
        user32.CallNextHookEx.restype = ct.c_ssize_t
        user32.UnhookWindowsHookEx.argtypes = [wt.HANDLE]
        user32.PeekMessageW.argtypes = [ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
        user32.DispatchMessageW.argtypes = [ct.POINTER(wt.MSG)]
        user32.DispatchMessageW.restype = ct.c_ssize_t
        user32.GetAsyncKeyState.argtypes = [ct.c_int]
        user32.GetAsyncKeyState.restype = ct.c_short
        kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wt.HMODULE

        class KeyboardData(ct.Structure):
            _fields_ = [("vkCode", wt.DWORD), ("scanCode", wt.DWORD), ("flags", wt.DWORD),
                        ("time", wt.DWORD), ("extra", ct.c_size_t)]

        class MouseData(ct.Structure):
            _fields_ = [("pt", wt.POINT), ("mouseData", wt.DWORD), ("flags", wt.DWORD),
                        ("time", wt.DWORD), ("extra", ct.c_size_t)]

        @hook_proc
        def keyboard(code, message, pointer):
            if code == 0:
                data = ct.cast(pointer, ct.POINTER(KeyboardData)).contents
                if not data.flags & 0x10:  # Do not treat synthesized input as user speech intent.
                    self.key_edge(data.vkCode, message in (0x100, 0x104))
            return user32.CallNextHookEx(None, code, message, pointer)

        @hook_proc
        def mouse(code, message, pointer):
            if code == 0 and message in (0x20B, 0x20C):
                data = ct.cast(pointer, ct.POINTER(MouseData)).contents
                if not data.flags & 1:
                    self.key_edge(5 if data.mouseData >> 16 == 1 else 6, message == 0x20B)
            return user32.CallNextHookEx(None, code, message, pointer)

        hooks = []
        try:
            for kind, callback in ((13, keyboard), (14, mouse)):
                hook = user32.SetWindowsHookExW(kind, callback, kernel32.GetModuleHandleW(None), 0)
                if not hook:
                    raise ct.WinError(ct.get_last_error())
                hooks.append(hook)
            self._ready.set()
            message = wt.MSG()
            while not self._stop.is_set():
                while user32.PeekMessageW(ct.byref(message), None, 0, 0, 1):
                    user32.TranslateMessage(ct.byref(message))
                    user32.DispatchMessageW(ct.byref(message))
                # Release reconciliation runs outside hooks, where async state is current.
                for key in tuple(self._pressed):
                    if not user32.GetAsyncKeyState(key) & 0x8000:
                        self.key_edge(key, False)
                self._stop.wait(0.005)
        except Exception as exc:
            self.error = str(exc)
        finally:
            for hook in hooks:
                user32.UnhookWindowsHookEx(hook)
            self._ready.set()
