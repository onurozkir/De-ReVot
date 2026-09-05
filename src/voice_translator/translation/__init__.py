"""Translation Adapter Layer."""

from voice_translator.translation.base import MTAdapter
from voice_translator.translation.ctranslate_backend import CTranslate2MTAdapter
from voice_translator.translation.mock_backend import MockMTAdapter

__all__ = ["MTAdapter", "CTranslate2MTAdapter", "MockMTAdapter"]

