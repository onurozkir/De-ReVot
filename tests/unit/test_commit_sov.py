from voice_translator.streaming.commit_policy import (
    CommitController,
    is_turkish_predicate_tail,
    is_open_conjunction_tail,
)


def test_is_turkish_predicate_tail_positives():
    predicates = [
        "Toplantı bitti",
        "Raporu hazırlıyorum",
        "Yarın görüşeceğiz",
        "Bunu yaptık",
        "Sorun yok",
        "Her şey hazır",
        "Bu doğru değil",
        "Geldiler mi",
        "Başlayabiliriz",
        "Gitmeli",
        "Hemen bakalım",
        "Sistemi kurdum",
        "Anladım",
    ]
    for text in predicates:
        assert is_turkish_predicate_tail(text), f"Expected predicate tail for: '{text}'"


def test_is_turkish_predicate_tail_negatives():
    non_predicates = [
        "Bugün hava çok",
        "Raporun detayları",
        "Toplantı ve",
        "Bir iki üç",
        "Şirkette",
        "Bizim için",
    ]
    for text in non_predicates:
        assert not is_turkish_predicate_tail(text), f"Did not expect predicate tail for: '{text}'"


def test_is_open_conjunction_tail():
    conjunctions = [
        "Toplantı bitti çünkü",
        "Geldik ama",
        "Bunu yaptık ve",
        "Fakat",
        "Lakin",
    ]
    for text in conjunctions:
        assert is_open_conjunction_tail(text), f"Expected conjunction tail for: '{text}'"

    assert not is_open_conjunction_tail("Toplantı bitti")
    assert not is_open_conjunction_tail("Rapor hazır")


def test_adaptive_sov_commit_on_predicate_silence():
    ctrl = CommitController(
        min_words=2,
        enable_adaptive_sov=True,
        sov_min_silence_ms=200,
    )
    
    # Not enough silence (100ms < 200ms)
    dec = ctrl.evaluate("Raporu hazırladım", silence_ms=100.0, language="tr")
    assert not dec.should_commit

    # Enough silence (250ms >= 200ms) on Turkish predicate
    dec = ctrl.evaluate("Raporu hazırladım", silence_ms=250.0, language="tr")
    assert dec.should_commit
    assert dec.reason == "turkish_sov_verb"
    assert dec.committed_text == "Raporu hazırladım"
    assert dec.remaining_partial_text == ""


def test_adaptive_sov_does_not_commit_on_incomplete_clause():
    ctrl = CommitController(
        min_words=2,
        enable_adaptive_sov=True,
        sov_min_silence_ms=200,
    )
    
    # 300ms silence but sentence ends with an open noun/modifier
    dec = ctrl.evaluate("Bugün şirketteki toplantı", silence_ms=300.0, language="tr")
    assert not dec.should_commit


def test_adaptive_sov_holds_on_conjunction_tail():
    ctrl = CommitController(
        min_words=2,
        enable_adaptive_sov=True,
        sov_min_silence_ms=200,
    )
    
    # Silence is 350ms, but user ended with conjunction 'çünkü'
    dec = ctrl.evaluate("Raporu hazırladım çünkü", silence_ms=350.0, language="tr")
    assert not dec.should_commit


def test_adaptive_sov_disabled_for_non_turkish():
    ctrl = CommitController(
        min_words=2,
        enable_adaptive_sov=True,
        sov_min_silence_ms=200,
    )
    
    # Even if predicate matches Turkish pattern or English text, language="en" does not trigger sov
    dec = ctrl.evaluate("I did it", silence_ms=300.0, language="en")
    assert not dec.should_commit or dec.reason != "turkish_sov_verb"

