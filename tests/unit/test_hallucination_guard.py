import pytest

from voice_translator.streaming.hallucination_guard import HallucinationGuard, SpeechEvidence, normalize_text


GOOD_EVIDENCE = SpeechEvidence(utterance_ms=1800, voiced_ms=1400, voiced_ratio=0.77, max_queue_age_ms=30)


@pytest.mark.parametrize("text", ["Teşekkür ederim", "Abone ol.", "Altyazı M.K.",
    "İzlediğiniz için teşekkür ederim.", "Thank you for watching!", "Görüşmek üzere",
    "Bugün altyazı hakkında konuşalım", "We need to review the deployment"])
def test_genuine_speech_is_never_rejected_by_its_words(text):
    assert HallucinationGuard().evaluate(text, GOOD_EVIDENCE, {"avg_logprob": -0.2}).accepted


def test_turkish_dotted_i_normalization_preserves_letter_counts():
    assert normalize_text("İzlediğiniz için teşekkür ederim.") == "izlediğiniz için teşekkür ederim"


@pytest.mark.parametrize("text", ["Teşekkür ederim", "Abone ol", "Altyazı M.K.", "arbitrary output"])
def test_silence_rejected_independently_of_words(text):
    evidence = SpeechEvidence(utterance_ms=600, voiced_ms=0, voiced_ratio=0)
    assert HallucinationGuard().evaluate(text, evidence).reason == "insufficient_voiced_audio"


def test_acoustically_impossible_text_is_rejected_but_genuine_duration_passes():
    text = "İzlediğiniz için teşekkür ederim"
    short = SpeechEvidence(utterance_ms=600, voiced_ms=250, voiced_ratio=0.42)
    assert HallucinationGuard().evaluate(text, short).reason == "implausible_speech_rate"
    assert HallucinationGuard().evaluate(text, GOOD_EVIDENCE).accepted


def test_punctuation_does_not_inflate_speech_rate():
    evidence = SpeechEvidence(utterance_ms=400, voiced_ms=300, voiced_ratio=0.75)
    assert HallucinationGuard().evaluate("Evet... Evet!", evidence).accepted


def test_whisper_metadata_still_rejects_non_speech_and_repetition():
    guard = HallucinationGuard()
    assert guard.evaluate("apparently valid", GOOD_EVIDENCE, {"no_speech_prob": 0.95}).reason == "whisper_no_speech"
    assert guard.evaluate("apparently valid", GOOD_EVIDENCE, {"compression_ratio": 3}).reason == "whisper_repetition"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "not-a-number"])
def test_invalid_model_confidence_cannot_bypass_guard(invalid):
    assert not HallucinationGuard().evaluate("Merhaba", GOOD_EVIDENCE, {"avg_logprob": invalid}).accepted
