import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "bert_model"
MAX_LENGTH = 128

# Articles sampled per class. The full ISOT dataset has ~21k fake / ~21k real;
# a subset keeps CPU fine-tuning under an hour without hurting accuracy much.
N_PER_CLASS = int(os.getenv("N_PER_CLASS", 5000))

# ---------------- LOAD DATA ----------------
# Kaggle "Fake and Real News" (ISOT) dataset:
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0  # FAKE
true["label"] = 1  # REAL

df = pd.concat([
    fake.sample(N_PER_CLASS, random_state=42),
    true.sample(N_PER_CLASS, random_state=42),
])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

texts = df["text"].tolist()
labels = df["label"].tolist()

# ---------------- SPLIT: 70% train / 15% val / 15% held-out test ----------------
train_texts, rest_texts, train_labels, rest_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    rest_texts, rest_labels, test_size=0.5, random_state=42, stratify=rest_labels
)

# The test split is never seen during training; evaluate.py scores against it.
pd.DataFrame({"text": test_texts, "label": test_labels}).to_csv(
    "test_split.csv", index=False
)
print(f"Split: {len(train_texts)} train / {len(val_texts)} val / {len(test_texts)} test")

# ---------------- TOKENIZER ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

# ---------------- DATASET ----------------
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)

# ---------------- MODEL ----------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "FAKE", 1: "REAL"},
    label2id={"FAKE": 0, "REAL": 1},
)

# ---------------- TRAIN ----------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# ---------------- SAVE ----------------
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"DistilBERT fine-tuned and saved to {MODEL_DIR}/")
print("Next: python evaluate.py  (reports test-set metrics)")
