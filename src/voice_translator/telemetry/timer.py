"""High-resolution monotonic timer utilities."""

import time


class MonotonicTimer:
    """Context manager and stopwatch using time.monotonic_ns()."""

    def __init__(self):
        self.start_ns: int = 0
        self.end_ns: int = 0

    def __enter__(self):
        self.start_ns = time.monotonic_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_ns = time.monotonic_ns()

    @property
    def elapsed_ms(self) -> float:
        end = self.end_ns if self.end_ns > 0 else time.monotonic_ns()
        return (end - self.start_ns) / 1e6

    @property
    def elapsed_ns(self) -> int:
        end = self.end_ns if self.end_ns > 0 else time.monotonic_ns()
        return end - self.start_ns

