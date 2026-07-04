"""Evaluate the fine-tuned model on the held-out test split.

Run after train_bert.py. Prints accuracy / precision / recall / F1,
saves them to metrics.json, and writes a confusion matrix image to demo/.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "bert_model"
TEST_FILE = "test_split.csv"
MAX_LENGTH = 128
BATCH_SIZE = 32

if not os.path.isdir(MODEL_DIR):
    raise SystemExit(f"Model not found at '{MODEL_DIR}'. Run train_bert.py first.")
if not os.path.exists(TEST_FILE):
    raise SystemExit(f"'{TEST_FILE}' not found. Run train_bert.py first.")

df = pd.read_csv(TEST_FILE)
texts = df["text"].astype(str).tolist()
y_true = df["label"].tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

y_pred = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(
            texts[i : i + BATCH_SIZE],
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        logits = model(**enc).logits
        y_pred.extend(logits.argmax(dim=-1).tolist())
        print(f"\r{min(i + BATCH_SIZE, len(texts))}/{len(texts)}", end="")
print()

# FAKE (label 0) is the positive class: precision = of predicted-fake, how
# many were fake; recall = of actual-fake, how many we caught.
metrics = {
    "test_samples": len(y_true),
    "accuracy": round(accuracy_score(y_true, y_pred), 4),
    "precision_fake": round(precision_score(y_true, y_pred, pos_label=0), 4),
    "recall_fake": round(recall_score(y_true, y_pred, pos_label=0), 4),
    "f1_fake": round(f1_score(y_true, y_pred, pos_label=0), 4),
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

cm = confusion_matrix(y_true, y_pred)
os.makedirs("demo", exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
disp.plot(cmap="Blues", colorbar=False)
plt.title("Confusion matrix — held-out test set")
plt.tight_layout()
plt.savefig("demo/confusion_matrix.png", dpi=150)

print(json.dumps(metrics, indent=2))
print("\nConfusion matrix [rows=actual FAKE/REAL, cols=predicted]:")
print(cm)
print("\nSaved metrics.json and demo/confusion_matrix.png")
