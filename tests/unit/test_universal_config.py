from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from voice_translator.config.models import AppConfig, StreamingConfig
from voice_translator.config.presets import APP_PRESETS, resolve_input_mode
from voice_translator.streaming.pipeline_runtime import build_guard


@pytest.mark.parametrize("preset", list(APP_PRESETS))
def test_presets_keep_explicit_user_mode_overrides(preset):
    assert resolve_input_mode(preset, "ptt") == "ptt"
    assert resolve_input_mode(preset, "vad") == "vad"


def test_game_presets_default_to_ptt_and_meetings_to_vad():
    assert resolve_input_mode("dota2", "auto") == "ptt"
    assert resolve_input_mode("google_meet", "auto") == "vad"


def test_speech_rate_setting_reaches_the_runtime_guard():
    config = StreamingConfig(guard_max_chars_per_second=42)
    assert build_guard(config).policy.max_chars_per_second == 42
    with pytest.raises(ValidationError):
        StreamingConfig(guard_max_chars_per_second=0)


def test_ct2_and_turbo_are_valid_configuration_options():
    config = AppConfig.model_validate({"asr": {"backend": "faster_whisper", "model_path": "models/asr/whisper-large-v3", "compute_type": "int8_float16"}})
    assert config.asr.compute_type == "int8_float16"
    assert AppConfig().asr.backend == "whisper_turbo"


def test_new_environment_prefix_takes_precedence_over_legacy(monkeypatch, tmp_path):
    from voice_translator.config.loader import load_config
    monkeypatch.setenv("TEAMS_TRANSLATOR_SERVER_PORT", "8111")
    monkeypatch.setenv("VOICE_TRANSLATOR_SERVER_PORT", "8222")
    assert load_config(project_root=str(tmp_path)).server.port == 8222


def test_language_registry_configuration_is_loaded():
    from voice_translator.config.loader import load_config
    config = load_config()
    assert config.languages.default_source == "tr"
    assert config.languages.default_target == "en"
    assert set(config.languages.enabled) == {"tr", "en", "fr"}
    assert config.languages.definitions["fr"].nllb_code == "fra_Latn"
    assert config.translation.pairs["tr-en"].endswith("opus-mt-tc-big-tr-en")


def test_language_registry_accepts_partial_toml_override():
    config = AppConfig.model_validate({
        "languages": {
            "enabled": ["en", "de"],
            "default_source": "en",
            "default_target": "de",
            "definitions": {
                "en": {"name": "English", "whisper_code": "en", "nllb_code": "eng_Latn"},
                "de": {"name": "Deutsch", "whisper_code": "de", "nllb_code": "deu_Latn"},
            },
        }
    })
    assert config.languages.definitions["de"].xtts_supported is True
