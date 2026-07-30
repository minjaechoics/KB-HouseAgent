"""ANSI console that keeps a live metric in the top three rows."""
from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable


def _safe_console_text(value: object) -> str:
    """Keep Windows CP949 consoles alive while preserving UTF-8 JSON reports.

    Some benchmark questions contain mathematical symbols such as ``¬`` that
    CP949 cannot encode.  Only the terminal rendering is replaced; in-memory
    values and saved reports remain unchanged.
    """
    text = str(value)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (LookupError, UnicodeEncodeError):
        return text.encode(encoding, errors="replace").decode(
            encoding, errors="replace")


class FixedHeaderConsole:
    def __init__(self, metric: Callable[[], str], refresh_seconds: float = 0.05):
        self.metric = metric
        self.refresh_seconds = refresh_seconds
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ansi = bool(getattr(sys.stdout, "isatty", lambda: False)())

    def _enable_windows_ansi(self) -> None:
        if os.name != "nt" or not self._ansi:
            return
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            handle = kernel.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            self._ansi = False

    def _paint_header(self) -> None:
        if not self._ansi:
            return
        with self._lock:
            # Save the scrolling cursor, repaint only rows 1..3, then restore it.
            sys.stdout.write(
                "\x1b7\x1b[1;1H\x1b[2K====================\n"
                f"\x1b[2K{_safe_console_text(self.metric())}\n"
                "\x1b[2K====================\x1b8"
            )
            sys.stdout.flush()

    def start(self) -> None:
        self._enable_windows_ansi()
        with self._lock:
            if self._ansi:
                sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write("====================\n")
            sys.stdout.write(_safe_console_text(self.metric()) + "\n")
            sys.stdout.write("====================\n\n")
            sys.stdout.flush()
        if self._ansi:
            self._thread = threading.Thread(target=self._ticker, daemon=True)
            self._thread.start()

    def _ticker(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            self._paint_header()

    def log(self, value: str = "") -> None:
        with self._lock:
            sys.stdout.write(_safe_console_text(value) + "\n")
            sys.stdout.flush()
        self._paint_header()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
        self._paint_header()


def format_elapsed(seconds: float) -> str:
    total_ms = max(0, int(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{millis:03d}"


def wait_for_key() -> None:
    print("\n아무 키나 누르면 종료합니다...", flush=True)
    if os.name == "nt" and sys.stdin.isatty():
        import msvcrt
        msvcrt.getwch()
    else:
        input()
