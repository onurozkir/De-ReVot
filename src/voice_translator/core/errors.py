"""Typed domain exceptions for Teams Translator."""


class TranslatorError(Exception):
    """Base exception for all translator domain errors."""


class AudioDeviceError(TranslatorError):
    """Raised on device discovery, opening, or WASAPI stream failure."""


class ModelNotFoundError(TranslatorError):
    """Raised when an explicit model path does not exist locally (no silent download)."""


class WarmupError(TranslatorError):
    """Raised when model warmup fails before reaching Ready state."""


class OverloadError(TranslatorError):
    """Raised when bounded queues reach hard capacity."""


class PersistenceError(TranslatorError):
    """Raised when database operations fail."""

