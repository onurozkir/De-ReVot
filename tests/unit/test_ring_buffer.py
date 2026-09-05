import numpy as np
import pytest
from voice_translator.core.ring_buffer import PCMRingBuffer


def test_ring_buffer_write_read():
    rb = PCMRingBuffer(capacity_samples=1000)
    data = np.ones(500, dtype=np.float32)
    
    written = rb.write(data)
    assert written == 500
    assert rb.available_read == 500
    assert rb.available_write == 500

    read_data = rb.read(300)
    assert len(read_data) == 300
    assert rb.available_read == 200
    np.testing.assert_array_equal(read_data, np.ones(300, dtype=np.float32))

    # Read remaining
    rem_data = rb.read()
    assert len(rem_data) == 200
    assert rb.available_read == 0


def test_ring_buffer_overflow():
    rb = PCMRingBuffer(capacity_samples=500)
    data = np.arange(600, dtype=np.float32)
    
    rb.write(data)
    assert rb.available_read == 500
    assert rb.overrun_count > 0

    read_data = rb.read()
    assert len(read_data) == 500
    # Should contain the most recent 500 samples
    np.testing.assert_array_equal(read_data, np.arange(100, 600, dtype=np.float32))


def test_ring_buffer_underrun():
    rb = PCMRingBuffer(capacity_samples=500)
    read_data = rb.read(100)
    assert len(read_data) == 0
    assert rb.underrun_count == 1

