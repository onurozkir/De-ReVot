from voice_translator.streaming.commit_policy import CommitController


def test_commit_punctuation():
    ctrl = CommitController(min_words=2)
    dec = ctrl.evaluate("Merhaba nasılsınız? Bugün hava çok güzel.")
    assert dec.should_commit
    assert dec.committed_text == "Merhaba nasılsınız?"
    assert dec.remaining_partial_text == "Bugün hava çok güzel."
    assert dec.reason == "punctuation"


def test_commit_silence_endpoint():
    ctrl = CommitController()
    dec = ctrl.evaluate("Toplantı başladı ve devam ediyor", is_silence_endpoint=True)
    assert dec.should_commit
    assert dec.committed_text == "Toplantı başladı ve devam ediyor"
    assert dec.reason == "silence_endpoint"


def test_commit_deadline_timeout():
    ctrl = CommitController(min_words=3, max_wait_ms=1000)
    # First partial
    _ = ctrl.evaluate("Bir iki üç dört beş", now_ms=1000.0)
    # After timeout
    dec = ctrl.evaluate("Bir iki üç dört beş", now_ms=2100.0)
    assert dec.should_commit
    assert dec.reason == "deadline_timeout"


def test_stable_prefix_matching_is_case_insensitive():
    ctrl = CommitController(min_words=3, max_wait_ms=10_000, stable_prefix_min_count=2)

    ctrl.evaluate("İstanbul dunya bugun eski", now_ms=1000.0)
    ctrl.evaluate("istanbul dunya bugun yeni", now_ms=1100.0)
    ctrl.evaluate("İSTANBUL DUNYA BUGUN sonra", now_ms=1200.0)

    assert ctrl.stable_prefix_matches == 2


def test_deadline_commits_only_stable_prefix_and_keeps_unstable_tail():
    ctrl = CommitController(min_words=3, max_wait_ms=1000, stable_prefix_min_count=2)

    ctrl.evaluate("Bir iki uc eski", now_ms=1000.0)
    ctrl.evaluate("Bir iki uc yeni", now_ms=1100.0)
    dec = ctrl.evaluate("Bir iki uc son", now_ms=2100.0)

    assert dec.should_commit
    assert dec.committed_text.casefold() == "bir iki uc"
    assert dec.remaining_partial_text == "son"
    assert dec.reason == "deadline_timeout"


def test_deadline_does_not_commit_when_no_stable_text_or_clause_exists():
    ctrl = CommitController(min_words=3, max_wait_ms=1000)

    ctrl.evaluate("Bir iki uc", now_ms=1000.0)
    dec = ctrl.evaluate("Dort bes alti", now_ms=2100.0)

    assert not dec.should_commit
    assert dec.committed_text == ""
    assert dec.remaining_partial_text == "Dort bes alti"


def test_deadline_can_commit_the_last_complete_clause_without_a_stable_prefix():
    ctrl = CommitController(min_words=3, max_wait_ms=1000)

    ctrl.evaluate("Tamamen farkli bir hipotez", now_ms=1000.0)
    dec = ctrl.evaluate("Bugun isi tamamladik ve yeni bolum belirsiz", now_ms=2100.0)

    assert dec.should_commit
    assert dec.committed_text == "Bugun isi tamamladik"
    assert dec.remaining_partial_text == "ve yeni bolum belirsiz"
    assert dec.reason == "deadline_timeout"
