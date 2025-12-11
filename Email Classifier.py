import pandas as pd
df=pd.read_csv("email_data.csv")
print(df)
"""
Merged & Fixed: Advanced Email Classifier
- Combines the simple cleaning sample and the advanced training pipeline.
- Fixes main-guard bug and small issues.
Save as: train_email_classifier.py
Run: python train_email_classifier.py
"""

import re
import string
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

# -----------------------------
# 1) NLTK setup (run once; quiet to reduce output)
# -----------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# -----------------------------
# 2) Text cleaning (robust + simple sample rules)
# -----------------------------
def clean_email(text: str) -> str:
    """
    Robust cleaning pipeline:
      - lowercase
      - remove emails, urls, html tags
      - remove numbers & punctuation
      - tokenize -> remove stopwords -> lemmatize
    """
    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # remove email addresses (fixed regex from simple file)
    text = re.sub(r'\S+@\S+', ' ', text)

    # remove urls
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # remove html tags
    text = re.sub(r'<.*?>', ' ', text)

    # remove numbers (optional)
    text = re.sub(r'\d+', ' ', text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize + stopword removal + lemmatize
    tokens = text.split()
    processed = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOP_WORDS]

    return " ".join(processed)

# -----------------------------
# 3) Load dataset (auto-detect column names)
# -----------------------------
def load_dataset(csv_path: str):
    """
    Loads CSV and returns DataFrame with columns ['email_text', 'label'].
    If CSV not found, returns a small demo dataframe.
    """
    if not os.path.exists(csv_path):
        print(f"[warning] CSV not found at {csv_path}. Using small demo dataset.")
        # small demo fallback (from simple file)
        emails = [
            "My internet is not working, I need help immediately",
            "I want to know the status of my refund request",
            "Great service! I appreciate the quick support",
            "You guys keep sending too many mails. Stop it."
        ]
        labels = ["complaint", "request", "feedback", "spam"]
        df = pd.DataFrame({"email_text": emails, "label": labels})
        return df

    df = pd.read_csv(csv_path)
    if df.shape[0] == 0:
        raise ValueError("CSV appears empty")

    # prefer category column if present (your original CSV used 'category')
    if "category" in df.columns and "email_text" in df.columns:
        df = df[["email_text", "category"]].dropna().reset_index(drop=True)
        df = df.rename(columns={"category": "label"})
    elif "label" in df.columns and "email_text" in df.columns:
        df = df[["email_text", "label"]].dropna().reset_index(drop=True)
    else:
        # try to salvage if subject + body present (simple file had 'subject' + 'body')
        if "body" in df.columns and "label" in df.columns:
            df["email_text"] = df["body"].astype(str)
            df = df[["email_text", "label"]].dropna().reset_index(drop=True)
        else:
            raise ValueError("CSV must contain 'email_text' and 'label' or 'category' column (or 'body' + 'label').")

    return df

# -----------------------------
# 4) Vectorize (TF-IDF w/ ngrams)
# -----------------------------
def vectorize_text(train_texts, test_texts,
                   ngram_range=(1, 2), max_features=8000, min_df=2):
    vect = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        norm='l2'
    )
    X_train = vect.fit_transform(train_texts)
    X_test = vect.transform(test_texts)
    return vect, X_train, X_test

# -----------------------------
# 5) Model training & selection (baseline)
# -----------------------------
def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "LinearSVC": LinearSVC(max_iter=2000),
        "NaiveBayes": MultinomialNB(),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1)
    }

    results = {}
    print("Baseline model comparison:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"  {name:15s} -> Accuracy: {acc:.4f}, F1 (weighted): {f1:.4f}")
        results[name] = model
    return results

# -----------------------------
# 6) Hyperparameter tuning (GridSearch examples)
# -----------------------------
def tune_logistic(X_train, y_train):
    param_grid = {
        "C": [0.1, 1.0, 5.0],
        "solver": ["lbfgs"],
        "penalty": ["l2"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(LogisticRegression(max_iter=2000),
                        param_grid,
                        cv=cv, n_jobs=-1, scoring='f1_weighted', verbose=0)
    grid.fit(X_train, y_train)
    print("Logistic best:", grid.best_params_, "score:", grid.best_score_)
    return grid.best_estimator_

def tune_svm(X_train, y_train):
    param_grid = {"C": [0.1, 1.0, 5.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(LinearSVC(max_iter=2000),
                        param_grid,
                        cv=cv, n_jobs=-1, scoring='f1_weighted', verbose=0)
    grid.fit(X_train, y_train)
    print("SVM best:", grid.best_params_, "score:", grid.best_score_)
    return grid.best_estimator_

# -----------------------------
# 7) Evaluate and print reports
# -----------------------------
def evaluate_model(model, X_test, y_test, label_names=None, plot_cm=True, save_plots=False):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=label_names) if label_names else confusion_matrix(y_test, preds)

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)

    if plot_cm:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(cm))
        if label_names is None:
            labels = sorted(list(set(y_test)))
        else:
            labels = label_names
        plt.xticks(tick_marks, labels, rotation=45, ha="right")
        plt.yticks(tick_marks, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        if save_plots:
            plt.savefig("confusion_matrix.png", bbox_inches='tight')
            print("Saved confusion_matrix.png")
        plt.show()

# -----------------------------
# 8) Save artifacts
# -----------------------------
def save_artifacts(model, vectorizer, model_path="best_email_model.pkl", vect_path="tfidf_vectorizer.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vect_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved vectorizer -> {vect_path}")

# -----------------------------
# 9) Predict helper
# -----------------------------
def predict_text(model, vectorizer, raw_text):
    cleaned = clean_email(raw_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return pred

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    csv_path = "email_data.csv"   # adjust if needed
    df = load_dataset(csv_path)
    print(f"Loaded {len(df)} rows.")

    # clean
    df["cleaned"] = df["email_text"].apply(clean_email)

    # show sample cleaned rows
    print("\nSample cleaned rows:")
    for i, row in df.head(5).iterrows():
        print(f" - RAW: {row['email_text']}")
        print(f"   CLEAN: {row['cleaned']}")

    # train/test split
    X = df["cleaned"]
    y = df["label"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y if len(set(y)) > 1 else None)

    # vectorize
    vectorizer, X_train, X_test = vectorize_text(X_train_text, X_test_text,
                                                 ngram_range=(1, 2),
                                                 max_features=8000,
                                                 min_df=2)

    # baseline comparison
    _ = compare_models(X_train, X_test, y_train, y_test)

    # hyperparameter tuning (example: logistic + svm)
    print("\nStarting hyperparameter tuning (Logistic + SVM). This may take a few minutes...")
    best_log = tune_logistic(X_train, y_train)
    best_svm = tune_svm(X_train, y_train)

    # evaluate the tuned models on test set
    print("\nEvaluation: Logistic (tuned)")
    evaluate_model(best_log, X_test, y_test, label_names=sorted(y.unique()), plot_cm=True)

    print("\nEvaluation: SVM (tuned)")
    evaluate_model(best_svm, X_test, y_test, label_names=sorted(y.unique()), plot_cm=True)

    # choose the best by weighted f1 on test (simple selection)
    def test_f1(model):
        p = model.predict(X_test)
        return f1_score(y_test, p, average='weighted')

    models_to_compare = {"Logistic": best_log, "SVM": best_svm}
    best_name, best_model = None, None
    best_score = -1
    for n, m in models_to_compare.items():
        sc = test_f1(m)
        print(f"{n} weighted-F1 on test: {sc:.4f}")
        if sc > best_score:
            best_score = sc
            best_name = n
            best_model = m

    print(f"\nSelected best model: {best_name} (weighted-F1={best_score:.4f})")

    # save best model + vectorizer
    save_artifacts(best_model, vectorizer,
                   model_path=f"best_email_model_{best_name}.pkl",
                   vect_path="tfidf_vectorizer.pkl")

    # demo predictions
    demo_examples = [
        "My order hasn't arrived and it's been 2 weeks.",
        "Claim your free iPad now by clicking here!",
        "Please send me the invoice for last month's order.",
        "Your app is amazing, thank you!"
    ]
    print("\nDemo predictions:")
    for text in demo_examples:
        pred = predict_text(best_model, vectorizer, text)
        print(f"  -> {text}  =>  {pred}")

    print("\nDone. Artifacts saved to disk.")

if __name__ == "__main__":
    main()
