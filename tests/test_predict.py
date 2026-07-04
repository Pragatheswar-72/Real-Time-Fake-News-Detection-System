"""Unit tests for the prediction function (model mocked — no checkpoint needed)."""

import fakenews


def _fake_pipeline(label, score):
    return lambda text: [{"label": label, "score": score}]


def test_predict_returns_real_label(monkeypatch):
    monkeypatch.setattr(fakenews, "_classifier", _fake_pipeline("REAL", 0.9731))
    label, score = fakenews.predict_news("Officials confirmed the new policy today.")
    assert label == "REAL"
    assert score == 97.31


def test_predict_returns_fake_label(monkeypatch):
    monkeypatch.setattr(fakenews, "_classifier", _fake_pipeline("FAKE", 0.881))
    label, score = fakenews.predict_news("Shocking secret they don't want you to know!")
    assert label == "FAKE"
    assert 0 <= score <= 100


def test_missing_model_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(fakenews, "_classifier", None)
    monkeypatch.setattr(fakenews, "MODEL_DIR", str(tmp_path / "nonexistent"))
    try:
        fakenews.predict_news("any text")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "train_bert.py" in str(e)
