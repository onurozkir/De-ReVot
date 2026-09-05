"""Lock-free/minimal-lock circular PCM ring buffer for audio callbacks."""

from __future__ import annotations

import threading
from typing import Optional, Tuple
import numpy as np


class PCMRingBuffer:
    """Fixed-capacity circular buffer for audio PCM data.
    
    Safe for non-blocking single-producer single-consumer operation.
    """

    def __init__(self, capacity_samples: int, channels: int = 1, dtype: np.dtype = np.float32):
        self.capacity = max(1, int(capacity_samples))
        self.channels = channels
        self.dtype = np.dtype(dtype)

        if channels == 1:
            self.buffer = np.zeros(self.capacity, dtype=self.dtype)
        else:
            self.buffer = np.zeros((self.capacity, channels), dtype=self.dtype)

        self._read_ptr = 0
        self._write_ptr = 0
        self._size = 0
        self._lock = threading.Lock()
        
        self.total_written_samples = 0
        self.total_read_samples = 0
        self.overrun_count = 0
        self.underrun_count = 0

    @property
    def available_read(self) -> int:
        with self._lock:
            return self._size

    @property
    def available_write(self) -> int:
        with self._lock:
            return self.capacity - self._size

    def write(self, data: np.ndarray) -> int:
        """Write samples into the ring buffer. Overwrites oldest on overflow."""
        data = np.asarray(data, dtype=self.dtype)
        num_samples = len(data)
        if num_samples == 0:
            return 0

        with self._lock:
            if num_samples > self.capacity:
                # Truncate to most recent capacity samples
                data = data[-self.capacity:]
                num_samples = self.capacity
                self.overrun_count += 1

            overflow = (self._size + num_samples) - self.capacity
            if overflow > 0:
                # Advance read pointer to discard overwritten samples
                self._read_ptr = (self._read_ptr + overflow) % self.capacity
                self._size -= overflow
                self.overrun_count += 1

            first_chunk = min(num_samples, self.capacity - self._write_ptr)
            second_chunk = num_samples - first_chunk

            self.buffer[self._write_ptr:self._write_ptr + first_chunk] = data[:first_chunk]
            if second_chunk > 0:
                self.buffer[:second_chunk] = data[first_chunk:]

            self._write_ptr = (self._write_ptr + num_samples) % self.capacity
            self._size += num_samples
            self.total_written_samples += num_samples
            return num_samples

    def read(self, max_samples: Optional[int] = None) -> np.ndarray:
        """Read available samples up to max_samples."""
        with self._lock:
            if self._size == 0:
                self.underrun_count += 1
                if self.channels == 1:
                    return np.empty(0, dtype=self.dtype)
                return np.empty((0, self.channels), dtype=self.dtype)

            to_read = self._size if max_samples is None else min(self._size, max_samples)
            first_chunk = min(to_read, self.capacity - self._read_ptr)
            second_chunk = to_read - first_chunk

            if self.channels == 1:
                out = np.empty(to_read, dtype=self.dtype)
            else:
                out = np.empty((to_read, self.channels), dtype=self.dtype)

            out[:first_chunk] = self.buffer[self._read_ptr:self._read_ptr + first_chunk]
            if second_chunk > 0:
                out[first_chunk:] = self.buffer[:second_chunk]

            self._read_ptr = (self._read_ptr + to_read) % self.capacity
            self._size -= to_read
            self.total_read_samples += to_read
            return out

    def peek(self, max_samples: Optional[int] = None) -> np.ndarray:
        """Inspect samples without advancing read pointer."""
        with self._lock:
            if self._size == 0:
                if self.channels == 1:
                    return np.empty(0, dtype=self.dtype)
                return np.empty((0, self.channels), dtype=self.dtype)

            to_read = self._size if max_samples is None else min(self._size, max_samples)
            first_chunk = min(to_read, self.capacity - self._read_ptr)
            second_chunk = to_read - first_chunk

            if self.channels == 1:
                out = np.empty(to_read, dtype=self.dtype)
            else:
                out = np.empty((to_read, self.channels), dtype=self.dtype)

            out[:first_chunk] = self.buffer[self._read_ptr:self._read_ptr + first_chunk]
            if second_chunk > 0:
                out[first_chunk:] = self.buffer[:second_chunk]
            return out

    def clear(self) -> None:
        with self._lock:
            self._read_ptr = 0
            self._write_ptr = 0
            self._size = 0
