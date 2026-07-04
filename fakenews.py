import os

from transformers import pipeline

MODEL_DIR = os.getenv("MODEL_DIR", "bert_model")

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError(
                f"Fine-tuned model not found at '{MODEL_DIR}'. "
                "Run `python train_bert.py` first (see README)."
            )
        _classifier = pipeline(
            "text-classification",
            model=MODEL_DIR,
            truncation=True,
            max_length=128,
        )
    return _classifier


def predict_news(text):
    """Classify news text as REAL or FAKE with a confidence percentage."""
    result = _get_classifier()(text)[0]
    return result["label"], round(result["score"] * 100, 2)
