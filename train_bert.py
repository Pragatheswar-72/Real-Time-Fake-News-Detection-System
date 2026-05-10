import pandas as pd
import torch
import os

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- LOAD DATA ----------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

# Reduce size (fast + stable)
df = pd.concat([
    fake.sample(10000, random_state=42),
    true.sample(10000, random_state=42)
])

df = df.sample(frac=1).reset_index(drop=True)

texts = df["text"].tolist()
labels = df["label"].tolist()

# ---------------- SPLIT ----------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ---------------- TOKENIZER ----------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

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
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ---------------- TRAIN ----------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# ---------------- SAVE ----------------
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

print("✅ BERT model trained and saved")