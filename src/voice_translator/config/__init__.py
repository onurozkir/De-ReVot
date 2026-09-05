"""Configuration module for Teams Translator."""

from voice_translator.config.loader import load_config
from voice_translator.config.models import AppConfig

__all__ = ["AppConfig", "load_config"]

