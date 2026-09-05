from pathlib import Path


APP_JS = Path(__file__).parents[2] / "src" / "voice_translator" / "web" / "static" / "app.js"


def test_ui_running_state_checks_use_only_the_normalized_status():
    script = APP_JS.read_text(encoding="utf-8")

    assert 'currentMeetingStatus === "RUNNING"' not in script
    assert 'if (status === "RUNNING")' not in script
    assert script.count('if (currentMeetingStatus.toLowerCase() === "running")') == 1
    assert script.count('if (norm === "running")') == 1


def test_ui_surfaces_precise_initialization_error():
    script = APP_JS.read_text(encoding="utf-8")

    assert "updateStatus(statData.status, statData.meeting_id, statData.error)" in script
    assert "meetingIdLabel.textContent = `Error: ${error}`" in script
