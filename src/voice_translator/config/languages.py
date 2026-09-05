"""Declarative language registry: enabled languages, model routing, XTTS matrix.

Language support is configuration-driven. Code outside this module must not
hardcode language codes, whisper codes, or MT pair model paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from voice_translator.config.models import AppConfig, LanguageDefinition

# XTTS-v2 synthesis matrix (Coqui docs). Definitions must not claim support
# outside this list; `xtts_supported` in the definition is the final gate.
XTTS_SUPPORTED = {
    "en", "tr", "fr", "de", "es", "it", "pt", "pl", "ru", "nl", "cs", "ar",
    "zh-cn", "ja", "hu", "ko", "hi",
}

# NLLB-200 FLORES codes for languages with registry definitions.
NLLB_FLORES: Dict[str, str] = {
    "tr": "tur_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "nl": "nld_Latn",
    "cs": "ces_Latn",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "hu": "hun_Latn",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
}


def definitions(config: AppConfig) -> Dict[str, LanguageDefinition]:
    return getattr(config, "languages").definitions


def enabled_codes(config: AppConfig) -> List[str]:
    langs = getattr(config, "languages")
    codes = [code for code in langs.enabled if code in langs.definitions]
    if codes:
        return codes
    return list(langs.definitions)


def normalize_code(code: str) -> str:
    return code.strip().lower().split("-")[0]


def get_definition(config: AppConfig, code: str) -> LanguageDefinition:
    normalized = normalize_code(code)
    lang_def = definitions(config).get(normalized)
    if lang_def is None:
        known = ", ".join(enabled_codes(config)) or "(none)"
        raise ValueError(
            f"Language '{code}' is not in the registry. Enabled languages: {known}. "
            "Add it under [languages.definitions.<code>] in config/default.toml."
        )
    return lang_def


def whisper_code(config: AppConfig, code: str) -> str:
    return get_definition(config, code).whisper_code


def nllb_code(config: AppConfig, code: str) -> str:
    return get_definition(config, code).nllb_code


def asr_prompt(config: AppConfig, code: str, fallback: str = "") -> str:
    lang_def = get_definition(config, code)
    return lang_def.asr_prompt.strip() or fallback


def xtts_supported(config: AppConfig, code: str) -> bool:
    lang_def = get_definition(config, code)
    return bool(lang_def.xtts_supported) and normalize_code(code) in XTTS_SUPPORTED


def defaults(config: AppConfig) -> tuple[str, str]:
    langs = getattr(config, "languages")
    source = normalize_code(langs.default_source)
    target = normalize_code(langs.default_target)
    known = enabled_codes(config)
    if source not in known and known:
        source = known[0]
    if target not in known and known:
        target = known[0]
    return source, target


def resolve_model_dir(path_str: str) -> Path:
    """Prefer a sibling -ct2 converted directory when it exists."""
    p = Path(path_str)
    if (p / "model.bin").exists():
        return p
    ct2_p = Path(f"{path_str}-ct2")
    if (ct2_p / "model.bin").exists():
        return ct2_p
    return p


def opus_pair_paths(config: AppConfig) -> Dict[str, str]:
    """Declared OPUS pair paths, with legacy fields as fallback bindings."""
    pairs: Dict[str, str] = dict(getattr(config.translation, "pairs", {}) or {})
    legacy = {
        "tr-en": config.translation.tr_en_model_path,
        "en-tr": config.translation.en_tr_model_path,
        "tr-fr": getattr(config.translation, "tr_fr_model_path", None),
    }
    for key, path in legacy.items():
        if key not in pairs and path:
            pairs[key] = path
    return pairs


def model_dir_usable(path_str: str) -> bool:
    """True when the directory contains actual model files, not just an empty folder."""
    p = resolve_model_dir(path_str)
    if not p.exists():
        return False
    return (p / "model.bin").exists() or (p / "config.json").exists()


def pair_installed(config: AppConfig, source: str, target: str) -> str:
    """Return 'opus', 'nllb' or '' for the routing of a language pair."""
    src, tgt = normalize_code(source), normalize_code(target)
    pairs = opus_pair_paths(config)
    opus_path = pairs.get(f"{src}-{tgt}")
    if opus_path and model_dir_usable(opus_path):
        return "opus"
    nllb_path = getattr(config.translation, "nllb_model_path", None)
    if nllb_path and model_dir_usable(nllb_path):
        return "nllb"
    return ""


def pair_download_hint(config: AppConfig, source: str, target: str) -> str:
    src, tgt = normalize_code(source), normalize_code(target)
    if f"{src}-{tgt}" in opus_pair_paths(config):
        return f"uv run python scripts/download_models.py mt-{src}-{tgt}"
    return "uv run python scripts/download_models.py mt-nllb-200"


def validate_pair(config: AppConfig, source: str, target: str, *, require_models: bool = False) -> None:
    """Validate a source/target pair; optionally require installed offline weights."""
    src, tgt = normalize_code(source), normalize_code(target)
    get_definition(config, src)
    get_definition(config, tgt)
    if require_models:
        route = pair_installed(config, src, tgt)
        if not route:
            raise ValueError(
                f"No offline MT model for {src}->{tgt}. "
                f"Download with: {pair_download_hint(config, src, tgt)}"
            )


def language_choices(config: AppConfig) -> List[dict]:
    """Web UI / API language list with pair availability."""
    pairs = opus_pair_paths(config)
    return [
        {
            "id": code,
            "name": definitions(config)[code].name,
            "whisper_code": definitions(config)[code].whisper_code,
            "xtts_supported": xtts_supported(config, code),
            "pairs": {
                other: (pair_installed(config, code, other) or None)
                for other in enabled_codes(config)
                if other != code
            },
        }
        for code in enabled_codes(config)
    ]


def validate_language(config: AppConfig, code: str) -> str:
    """Return the normalized enabled code or raise a clear ValueError."""
    normalized = normalize_code(code)
    if normalized not in enabled_codes(config):
        known = ", ".join(enabled_codes(config)) or "(none)"
        raise ValueError(f"Language '{code}' is not enabled. Enabled languages: {known}.")
    return normalized
