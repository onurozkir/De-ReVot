from voice_translator.core.types import Direction, LatencyEvent
from voice_translator.telemetry.metrics import TelemetryTracker


def test_telemetry_percentiles():
    tracker = TelemetryTracker(window_size=100)
    for dur in [100.0, 200.0, 300.0, 400.0, 500.0]:
        tracker.record_event(LatencyEvent(
            meeting_id="m1",
            utterance_id="u1",
            direction=Direction.INCOMING,
            event_type="incoming_partial",
            duration_ms=dur,
        ))

    p = tracker.get_percentiles("incoming_partial")
    assert p["count"] == 5
    assert p["p50"] == 300.0
    assert p["p95"] == 480.0


def test_asr_inference_wait_is_split_by_direction():
    tracker = TelemetryTracker(window_size=100)
    tracker.record_event(LatencyEvent(
        meeting_id="m1",
        utterance_id="tx_1",
        direction=Direction.OUTGOING,
        event_type="asr_inference_wait",
        duration_ms=10.0,
    ))
    tracker.record_event(LatencyEvent(
        meeting_id="m1",
        utterance_id="rx_1",
        direction=Direction.INCOMING,
        event_type="asr_inference_wait",
        duration_ms=20.0,
    ))

    snapshot = tracker.get_snapshot()
    assert snapshot["asr_wait_outgoing"]["count"] == 1
    assert snapshot["asr_wait_outgoing"]["p50"] == 10.0
    assert snapshot["asr_wait_incoming"]["count"] == 1
    assert snapshot["asr_wait_incoming"]["p50"] == 20.0
