"""Transparent, click-through Win32 subtitle HUD without a second UI framework."""

from collections import deque
import ctypes as ct
from ctypes import wintypes as wt
import sys
import threading
import time
import uuid


class SubtitleState:
    """Bounded projection: ordered commits plus one replaceable partial."""

    def __init__(self, expiry_sec: float = 8):
        self.expiry_sec = expiry_sec
        self._lock = threading.Lock()
        self.commits = deque(maxlen=2)
        self.partial = ""
        self.committed_sequence = -1
        self.partial_key = (-1, -1)
        self.updated_at = 0.0

    def update(self, event: dict, now: float | None = None) -> None:
        kind = event.get("type")
        if kind not in ("incoming_partial", "incoming_committed"):
            return
        sequence = int(event.get("sequence_id", 0))
        text = str(event.get("translated_text", ""))[:600]
        with self._lock:
            if sequence <= self.committed_sequence:
                return
            if kind == "incoming_partial":
                key = (sequence, int(event.get("revision", 0)))
                if key <= self.partial_key:
                    return
                self.partial_key = key
                self.partial = text
            else:
                self.commits.append(text)
                self.committed_sequence = sequence
                if sequence >= self.partial_key[0]:
                    self.partial = ""
            self.updated_at = time.monotonic() if now is None else now

    def text(self, now: float | None = None) -> str:
        with self._lock:
            if (time.monotonic() if now is None else now) - self.updated_at > self.expiry_sec:
                return ""
            # Show current partial and last commit; history remains in the primary UI.
            return "\n".join(filter(None, [self.commits[-1] if self.commits else "", self.partial]))[-600:]

    def clear(self) -> None:
        with self._lock:
            self.commits.clear()
            self.partial = ""
            self.committed_sequence = -1
            self.partial_key = (-1, -1)
            self.updated_at = 0


class DesktopOverlay:
    EX_STYLE = 0x00080000 | 0x00000020 | 0x00000008 | 0x08000000 | 0x00000080

    def __init__(self, config):
        self.config = config
        self.state = SubtitleState(config.expiry_sec)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self.error: str | None = None
        self.hwnd = None

    @property
    def running(self) -> bool:
        return bool(self.hwnd and self._thread and self._thread.is_alive())

    def start(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Desktop subtitles require native Windows.")
        if self.running:
            return
        self.error = None
        self.state.clear()
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, name="voice-subtitle-hud", daemon=True)
        self._thread.start()
        if not self._ready.wait(3) or self.error:
            self.stop()
            raise RuntimeError(self.error or "Desktop subtitle initialization timed out.")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.state.clear()

    def _run(self) -> None:
        user32 = ct.WinDLL("user32", use_last_error=True)
        gdi32 = ct.WinDLL("gdi32", use_last_error=True)
        kernel32 = ct.WinDLL("kernel32", use_last_error=True)
        proc_type = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

        class WindowClass(ct.Structure):
            _fields_ = [("style", wt.UINT), ("proc", proc_type), ("cls_extra", ct.c_int),
                        ("wnd_extra", ct.c_int), ("instance", wt.HINSTANCE), ("icon", wt.HICON),
                        ("cursor", wt.HANDLE), ("background", wt.HBRUSH),
                        ("menu", wt.LPCWSTR), ("name", wt.LPCWSTR)]

        class Paint(ct.Structure):
            _fields_ = [("dc", wt.HDC), ("erase", wt.BOOL), ("rect", wt.RECT),
                        ("restore", wt.BOOL), ("update", wt.BOOL), ("reserved", ct.c_byte * 32)]

        # Explicit pointer-width ABI is required on 64-bit Windows.
        signatures = [
            (kernel32, "GetModuleHandleW", [wt.LPCWSTR], wt.HMODULE),
            (user32, "RegisterClassW", [ct.POINTER(WindowClass)], wt.ATOM),
            (user32, "UnregisterClassW", [wt.LPCWSTR, wt.HINSTANCE], wt.BOOL),
            (user32, "CreateWindowExW", [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ct.c_int, ct.c_int, ct.c_int, ct.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, ct.c_void_p], wt.HWND),
            (user32, "DefWindowProcW", [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM], ct.c_ssize_t),
            (user32, "BeginPaint", [wt.HWND, ct.POINTER(Paint)], wt.HDC),
            (user32, "EndPaint", [wt.HWND, ct.POINTER(Paint)], wt.BOOL),
            (user32, "FillRect", [wt.HDC, ct.POINTER(wt.RECT), wt.HBRUSH], ct.c_int),
            (user32, "GetClientRect", [wt.HWND, ct.POINTER(wt.RECT)], wt.BOOL),
            (user32, "DrawTextW", [wt.HDC, wt.LPCWSTR, ct.c_int, ct.POINTER(wt.RECT), wt.UINT], ct.c_int),
            (user32, "SetLayeredWindowAttributes", [wt.HWND, wt.DWORD, wt.BYTE, wt.DWORD], wt.BOOL),
            (user32, "SetWindowPos", [wt.HWND, wt.HWND, ct.c_int, ct.c_int, ct.c_int, ct.c_int, wt.UINT], wt.BOOL),
            (user32, "ShowWindow", [wt.HWND, ct.c_int], wt.BOOL),
            (user32, "DestroyWindow", [wt.HWND], wt.BOOL),
            (user32, "InvalidateRect", [wt.HWND, ct.POINTER(wt.RECT), wt.BOOL], wt.BOOL),
            (user32, "PeekMessageW", [ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT], wt.BOOL),
            (user32, "DispatchMessageW", [ct.POINTER(wt.MSG)], ct.c_ssize_t),
            (gdi32, "CreateSolidBrush", [wt.DWORD], wt.HBRUSH),
            (gdi32, "CreateFontW", [ct.c_int] * 5 + [wt.DWORD] * 8 + [wt.LPCWSTR], wt.HFONT),
            (gdi32, "SelectObject", [wt.HDC, wt.HANDLE], wt.HANDLE),
            (gdi32, "SetBkMode", [wt.HDC, ct.c_int], ct.c_int),
            (gdi32, "SetTextColor", [wt.HDC, wt.DWORD], wt.DWORD),
            (gdi32, "DeleteObject", [wt.HANDLE], wt.BOOL),
        ]
        for library, name, args, result in signatures:
            function = getattr(library, name)
            function.argtypes, function.restype = args, result

        brush = font = None
        registered = False
        class_name = "VoiceTranslatorHUD_" + uuid.uuid4().hex
        instance = kernel32.GetModuleHandleW(None)
        displayed = ""

        @proc_type
        def window_proc(hwnd, message, wparam, lparam):
            if message == 0x84:  # WM_NCHITTEST, never eat a click or steal focus.
                return -1
            if message == 0x21:  # WM_MOUSEACTIVATE
                return 3
            if message == 0xF:
                paint = Paint()
                dc = user32.BeginPaint(hwnd, ct.byref(paint))
                try:
                    rect = wt.RECT()
                    user32.GetClientRect(hwnd, ct.byref(rect))
                    user32.FillRect(dc, ct.byref(rect), brush)
                    old_font = gdi32.SelectObject(dc, font)
                    gdi32.SetBkMode(dc, 1)
                    gdi32.SetTextColor(dc, 0x000000)
                    rect.left, rect.top, rect.right = 12, 12, rect.right - 8
                    user32.DrawTextW(dc, displayed, -1, ct.byref(rect), 0x811)
                    rect.left, rect.top, rect.right = 10, 10, rect.right - 2
                    gdi32.SetTextColor(dc, 0xFFFFFF)
                    user32.DrawTextW(dc, displayed, -1, ct.byref(rect), 0x811)
                    gdi32.SelectObject(dc, old_font)
                finally:
                    user32.EndPaint(hwnd, ct.byref(paint))
                return 0
            return user32.DefWindowProcW(hwnd, message, wparam, lparam)

        try:
            brush = gdi32.CreateSolidBrush(0x010101)
            font = gdi32.CreateFontW(-self.config.font_size, 0, 0, 0, 600, 0, 0, 0, 1, 0, 0, 3, 0, "Segoe UI")
            if not brush or not font:
                raise ct.WinError(ct.get_last_error())
            window_class = WindowClass(0, window_proc, 0, 0, instance, None, None, brush, None, class_name)
            if not user32.RegisterClassW(ct.byref(window_class)):
                raise ct.WinError(ct.get_last_error())
            registered = True
            screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            width = min(self.config.width, screen_width)
            height = min(240, max(140, self.config.font_size * 6))
            x, y = (screen_width - width) // 2, max(0, screen_height - height - self.config.bottom_margin)
            self.hwnd = user32.CreateWindowExW(self.EX_STYLE, class_name, "De-ReVot Subtitles", 0x80000000,
                                               x, y, width, height, None, None, instance, None)
            if not self.hwnd:
                raise ct.WinError(ct.get_last_error())
            if not user32.SetLayeredWindowAttributes(self.hwnd, 0x010101, 240, 3):
                raise ct.WinError(ct.get_last_error())
            user32.ShowWindow(self.hwnd, 4)  # SW_SHOWNOACTIVATE
            self._ready.set()
            message = wt.MSG()
            while not self._stop.is_set():
                while user32.PeekMessageW(ct.byref(message), None, 0, 0, 1):
                    user32.TranslateMessage(ct.byref(message))
                    user32.DispatchMessageW(ct.byref(message))
                text = self.state.text()
                if text != displayed:
                    displayed = text
                    user32.InvalidateRect(self.hwnd, None, True)
                    user32.SetWindowPos(self.hwnd, wt.HWND(-1), 0, 0, 0, 0, 0x13)
                self._stop.wait(0.03)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.hwnd:
                user32.DestroyWindow(self.hwnd)
                self.hwnd = None
            if registered:
                user32.UnregisterClassW(class_name, instance)
            for handle in (font, brush):
                if handle:
                    gdi32.DeleteObject(handle)
            self._ready.set()
