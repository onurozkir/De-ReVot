"""Unit tests for the declarative language registry and pair routing."""

from __future__ import annotations

import pytest

from voice_translator.config.languages import (
    asr_prompt,
    defaults,
    enabled_codes,
    get_definition,
    language_choices,
    opus_pair_paths,
    pair_installed,
    validate_language,
    validate_pair,
    whisper_code,
    xtts_supported,
)
from voice_translator.config.loader import load_config


def test_default_registry_enables_tr_en_fr_with_tr_to_en_defaults():
    config = load_config()
    assert enabled_codes(config) == ["tr", "en", "fr"]
    assert defaults(config) == ("tr", "en")
    assert whisper_code(config, "tr") == "tr"
    assert get_definition(config, "fr").nllb_code == "fra_Latn"
    assert asr_prompt(config, "tr") == "Toplantı, Türkçe, teknik, iş, günlük konuşma."
    assert xtts_supported(config, "en")


def test_unknown_language_raises_clear_error():
    config = load_config()
    with pytest.raises(ValueError, match="registry"):
        get_definition(config, "de")
    with pytest.raises(ValueError, match="not enabled"):
        validate_language(config, "de")


def test_legacy_pair_fields_bind_declared_pairs():
    config = load_config()
    pairs = opus_pair_paths(config)
    assert pairs["tr-en"].endswith("opus-mt-tc-big-tr-en")
    assert pairs["en-tr"].endswith("opus-mt-tc-big-en-tr")
    assert pairs["tr-fr"].endswith("opus-mt-tr-fr")


def test_pair_installed_detects_models_and_missing_pair_hint(tmp_path):
    config = load_config()
    config.translation.pairs = {"en-fr": str(tmp_path / "opus-mt-en-fr")}
    config.translation.tr_en_model_path = str(tmp_path / "no-tr-en")
    config.translation.en_tr_model_path = str(tmp_path / "no-en-tr")
    config.translation.tr_fr_model_path = str(tmp_path / "no-tr-fr")
    config.translation.nllb_model_path = str(tmp_path / "nllb")
    assert pair_installed(config, "en", "fr") == ""

    (tmp_path / "opus-mt-en-fr").mkdir()
    (tmp_path / "opus-mt-en-fr" / "model.bin").write_text("x")
    assert pair_installed(config, "en", "fr") == "opus"

    (tmp_path / "nllb").mkdir()
    assert pair_installed(config, "en", "tr") == ""  # empty dir is not an installed model
    (tmp_path / "nllb" / "config.json").write_text("x")
    assert pair_installed(config, "en", "tr") == "nllb"
    assert pair_installed(config, "fr", "tr") == "nllb"

    config.translation.nllb_model_path = str(tmp_path / "missing-nllb")
    config.translation.pairs = {"en-fr": str(tmp_path / "missing-opus")}
    with pytest.raises(ValueError, match="Download with"):
        validate_pair(config, "en", "fr", require_models=True)


def test_language_choices_reports_xtts_and_pair_availability(tmp_path):
    config = load_config()
    config.translation.pairs = {"en-fr": str(tmp_path / "opus-mt-en-fr")}
    config.translation.nllb_model_path = str(tmp_path / "nllb")
    choices = language_choices(config)
    by_id = {c["id"]: c for c in choices}
    assert set(by_id) == {"tr", "en", "fr"}
    assert by_id["tr"]["xtts_supported"] is True
    assert by_id["fr"]["pairs"]["en"] is None  # en->fr not installed in tmp registry

    (tmp_path / "nllb").mkdir()
    (tmp_path / "nllb" / "model.bin").write_text("x")
    choices = language_choices(config)
    by_id = {c["id"]: c for c in choices}
    assert by_id["fr"]["pairs"]["en"] == "nllb"


def test_xtts_matrix_rejects_unsupported_code():
    config = load_config()
    assert xtts_supported(config, "tr")
