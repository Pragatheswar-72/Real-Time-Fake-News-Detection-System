from transformers import pipeline

# Load lightweight model (no heavy training needed)
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict_news(text):
    result = classifier(text)[0]
    score = result["score"] * 100
    text_lower = text.lower()

    # ✅ REAL patterns (very important fix)
    if any(word in text_lower for word in [
        "reuters", "ap news", "bbc", "cnn",
        "according to", "reported", "reports",
        "official", "statement", "announced",
        "ministry", "government", "confirmed"
    ]):
        return "REAL", round(max(score, 75), 2)

    # ❌ FAKE patterns
    if any(word in text_lower for word in [
        "shocking", "100%", "secret", "leaked",
        "exposed", "hidden truth", "they don't want you to know"
    ]):
        return "FAKE", round(max(score, 80), 2)

    # 🎯 FINAL DECISION (ONLY REAL / FAKE)
    if result["label"] == "POSITIVE":
        return "REAL", round(score, 2)
    else:
        return "FAKE", round(score, 2)