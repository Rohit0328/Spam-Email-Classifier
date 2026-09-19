import os
import pandas as pd
import re
import string
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# --------------------------------------------------
# 1. NLTK Setup
# --------------------------------------------------

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

print("1. NLTK loaded")


# --------------------------------------------------
# 2. Dataset Paths
# --------------------------------------------------

hampath = "spam-ham/ham_zipped/main_ham"
spampath = "spam-ham/spam_zipped/main_spam"


# --------------------------------------------------
# 3. Load Dataset
# --------------------------------------------------

data = []


# Load Ham emails
for filename in os.listdir(hampath):
    filepath = os.path.join(hampath, filename)

    with open(filepath, "r", encoding="latin-1") as file:
        email = file.read()

        data.append({
            "text": email,
            "label": 0
        })


# Load Spam emails
for filename in os.listdir(spampath):
    filepath = os.path.join(spampath, filename)

    with open(filepath, "r", encoding="latin-1") as file:
        email = file.read()

        data.append({
            "text": email,
            "label": 1
        })


df = pd.DataFrame(data)

print("2. DataFrame created")
print("Shape:", df.shape)


# --------------------------------------------------
# 4. Text Cleaning
# --------------------------------------------------

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Replace email addresses
    text = re.sub(r"\S+@\S+", " EMAILADDRESS ", text)

    # Replace URLs
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Convert tokens back into text
    return " ".join(tokens)


df["text"] = df["text"].apply(clean_text)

print("3. Cleaning completed")

print("\n********* Clean Email ***************")
print(df["text"].iloc[0])


# --------------------------------------------------
# 5. Train/Test Split
# --------------------------------------------------

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 6. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# --------------------------------------------------
# 7. Logistic Regression
# --------------------------------------------------

logistic_model = LogisticRegression()

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_tfidf
)

print("\n===== Logistic Regression =====")
print(
    "Accuracy:",
    accuracy_score(y_test, logistic_prediction)
)

print(
    classification_report(
        y_test,
        logistic_prediction
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_prediction
    )
)


# --------------------------------------------------
# 8. Naive Bayes
# --------------------------------------------------

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_prediction = nb_model.predict(
    X_test_tfidf
)

print("\n===== Naive Bayes =====")
print(
    "Accuracy:",
    accuracy_score(y_test, nb_prediction)
)

print(
    classification_report(
        y_test,
        nb_prediction
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        nb_prediction
    )
)


# --------------------------------------------------
# 9. Linear SVM
# --------------------------------------------------

svm_model = LinearSVC()

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_prediction = svm_model.predict(
    X_test_tfidf
)

print("\n===== Linear SVM =====")
print(
    "Accuracy:",
    accuracy_score(y_test, svm_prediction)
)

print(
    classification_report(
        y_test,
        svm_prediction
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        svm_prediction
    )
)


# --------------------------------------------------
# 10. Save Final Model
# --------------------------------------------------

joblib.dump(
    svm_model,
    "spam_svm_model.pkl"
)

joblib.dump(
    vectorizer,
    "tfidf_vectorizer.pkl"
)

print(
    "\nFinal SVM model and vectorizer saved successfully!"
)