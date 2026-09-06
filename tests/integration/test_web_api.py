import pytest
from fastapi.testclient import TestClient
from voice_translator.config.loader import load_config
from voice_translator.streaming.orchestrator import MeetingOrchestrator
from voice_translator.web.server import create_app


def test_web_api_endpoints():
    config = load_config()
    orchestrator = MeetingOrchestrator(config=config, use_mocks=True)
    app = create_app(orchestrator)
    client = TestClient(app)

    # Test status endpoint
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "system" in data
    assert "error" in data

    # Test devices endpoint
    res = client.get("/api/devices")
    assert res.status_code == 200
    assert "devices" in res.json()
    for device in res.json()["devices"]:
        assert "stable_id" in device
        assert "host_api" in device
        assert "max_input_channels" in device
        assert "roles" in device

    res = client.get("/api/audio/diagnostics")
    assert res.status_code == 200
    diagnostics = res.json()
    assert "resolved" in diagnostics
    assert "outgoing" in diagnostics
    assert "incoming" in diagnostics

    # Test profiles endpoint
    res = client.get("/api/profiles")
    assert res.status_code == 200
    profiles = res.json()["profiles"]
    assert len(profiles) >= 1
    assert "target_languages" in profiles[0]
    assert profiles[0]["reference_count"] == len(profiles[0]["reference_audio_paths"])

    # Test language registry endpoint
    res = client.get("/api/languages")
    assert res.status_code == 200
    lang_data = res.json()
    assert {"tr", "en", "fr"} <= {l["id"] for l in lang_data["languages"]}
    assert lang_data["defaults"] == {"source": "tr", "target": "en"}
    assert lang_data["languages"][0]["xtts_supported"] is True

    # Test switch voice and language endpoints
    res = client.post("/api/meeting/switch_voice", json={"profile_id": profiles[0]["id"]})
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

    res = client.post("/api/meeting/switch_languages", json={"source_language": "en", "target_language": "fr"})
    assert res.status_code == 200
    assert res.json()["target_language"] == "fr"

    res = client.post("/api/meeting/switch_language", json={"target_language": "fr"})
    assert res.status_code == 200
    assert res.json()["target_language"] == "fr"

    res = client.post("/api/meeting/switch_language", json={"target_language": "unsupported"})
    assert res.status_code == 400

    res = client.post("/api/meeting/switch_languages", json={"source_language": "tr", "target_language": "de"})
    assert res.status_code == 400


def test_session_options_and_live_controls_keep_incoming_active_in_ptt():
    from types import SimpleNamespace
    from voice_translator.core.types import MeetingStatus
    config = load_config()
    instance = MeetingOrchestrator(config, use_mocks=True)
    instance.status = MeetingStatus.RUNNING
    outgoing_calls, incoming_pauses = [], []
    instance.outgoing_pipeline = SimpleNamespace(
        request_input=lambda *args: outgoing_calls.append(args),
        set_routing_muted=lambda value: outgoing_calls.append(("routing_muted", value)))
    instance.incoming_pipeline = SimpleNamespace(set_paused=incoming_pauses.append)
    client = TestClient(create_app(instance))
    options = client.get("/api/session/options").json()
    assert {"teams", "zoom", "google_meet", "discord", "slack", "dota2", "pubg", "cs2"} <= {p["id"] for p in options["presets"]}
    assert {"turbo", "large_v3", "configured"} == {m["id"] for m in options["asr_models"]}
    assert options["audio_processing"] == {"noise_suppression": True, "echo_cancellation": True}
    assert client.post("/api/session/controls", json={"input_mode": "ptt"}).status_code == 200
    assert client.post("/api/session/controls", json={"ptt_pressed": True}).json()["ptt_pressed"]
    assert client.post("/api/session/controls", json={"ptt_pressed": False}).status_code == 200
    assert incoming_pauses == [False, False, False]
    assert client.post("/api/session/controls", json={"muted": True}).json()["muted"]
    assert incoming_pauses[-1] is False
    assert client.post("/api/session/controls", json={"paused": True}).json()["paused"]
    assert incoming_pauses[-1] is True
    assert client.post("/api/session/controls", json={"input_mode": "invalid"}).status_code == 422
    instance.device_manager.close()


def test_controls_reject_requests_before_a_session():
    instance = MeetingOrchestrator(load_config(), use_mocks=True)
    client = TestClient(create_app(instance))
    assert client.post("/api/session/controls", json={"ptt_pressed": True}).status_code == 400
    instance.device_manager.close()


def test_start_api_forwards_explicit_audio_processing_choices(monkeypatch):
    from unittest.mock import AsyncMock
    instance = MeetingOrchestrator(load_config(), use_mocks=True)
    start = AsyncMock()
    monkeypatch.setattr(instance, "start_meeting", start)
    client = TestClient(create_app(instance))
    result = client.post("/api/meeting/start", json={"noise_suppression": False, "echo_cancellation": True})
    assert result.status_code == 200
    assert start.call_args.kwargs["noise_suppression"] is False
    assert start.call_args.kwargs["echo_cancellation"] is True
    instance.device_manager.close()
