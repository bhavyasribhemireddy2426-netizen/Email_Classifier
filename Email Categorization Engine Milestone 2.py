import pandas as pd
df=pd.read_csv("email_data.csv")
print(df)
# ============================================================
# Milestone 2: Email Classification Models & Evaluation
# ============================================================

import pandas as pd
import torch
print(torch.__version__)
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("email_data.csv")

texts = df["email_text"].tolist()
labels = df["category"].tolist()

# ----------------------------
# TF-IDF
# ----------------------------
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))

X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

# ============================================================
# STEP 1: Baseline Classifiers
# ============================================================

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\n=== Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# ============================================================
# STEP 2: Transformer Model (DistilBERT)
# ============================================================

category_map = {
    "complaint": 0,
    "request": 1,
    "spam": 2,
    "feedback": 3
}

numeric_category = [category_map[category] for category in category_map]

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128
)

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, category):
        self.encodings = encodings
        self.category = category

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["category"] = torch.tensor(self.category[idx])
        return item

    def __len__(self):
        return len(self.category)

dataset = EmailDataset(encodings, numeric_category)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)


# ----------------------------
# Prediction Demo
# ----------------------------
test_email = "Internet not working, please fix urgently"

inputs = tokenizer(test_email, return_tensors="pt")
outputs = model(**inputs)

predicted_class = torch.argmax(outputs.logits, dim=1)
reverse_category_map = {v: k for k, v in category_map.items()}

print("\nDistilBERT Prediction:")
print(reverse_category_map[predicted_class.item()])