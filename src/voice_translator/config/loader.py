"""Configuration loader supporting default TOML, local TOML override, and env vars."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Dict

from voice_translator.config.models import AppConfig


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_config(
    default_path: str = "config/default.toml",
    local_path: str = "config/local.toml",
    project_root: str = ".",
) -> AppConfig:
    root = Path(project_root).resolve()
    merged: Dict[str, Any] = {}

    # 1. Load default.toml
    def_file = root / default_path
    if def_file.exists():
        with open(def_file, "rb") as f:
            merged = tomllib.load(f)

    # 2. Load local.toml if exists
    loc_file = root / local_path
    if loc_file.exists():
        with open(loc_file, "rb") as f:
            loc_data = tomllib.load(f)
            _deep_update(merged, loc_data)

    # New product prefix wins; legacy prefix remains compatible during migration.
    overrides = [(prefix, key, value) for prefix in ("TEAMS_TRANSLATOR_", "VOICE_TRANSLATOR_")
                 for key, value in os.environ.items() if key.startswith(prefix)]
    for prefix, env_key, env_val in overrides:
        if env_key.startswith(prefix):
            parts = env_key[len(prefix):].lower().split("_", 1)
            if len(parts) == 2:
                section, key = parts
                if section not in merged:
                    merged[section] = {}
                # Auto convert numbers/booleans if possible
                if env_val.lower() in ("true", "1"):
                    merged[section][key] = True
                elif env_val.lower() in ("false", "0"):
                    merged[section][key] = False
                elif env_val.isdigit():
                    merged[section][key] = int(env_val)
                else:
                    merged[section][key] = env_val

    return AppConfig.model_validate(merged)
