import re
import string
import joblib
import nltk

from nltk.corpus import stopwords


# --------------------------------------------------
# 1. NLTK Setup
# --------------------------------------------------

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# --------------------------------------------------
# 2. Load Model and Vectorizer
# --------------------------------------------------

model = joblib.load("spam_svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")


# --------------------------------------------------
# 3. Text Cleaning
# --------------------------------------------------

def clean_text(text):

    # Lowercase
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

    return " ".join(tokens)


# --------------------------------------------------
# 4. Get Email
# --------------------------------------------------

email = input("\nEnter an email: ")


# --------------------------------------------------
# 5. Clean Email
# --------------------------------------------------

cleaned_email = clean_text(email)

print("\nCleaned email:")
print(cleaned_email)


# --------------------------------------------------
# 6. Convert Email to TF-IDF
# --------------------------------------------------

email_tfidf = vectorizer.transform(
    [cleaned_email]
)

print("\nTF-IDF conversion completed")
print("Shape:", email_tfidf.shape)


# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------

prediction = model.predict(email_tfidf)


# --------------------------------------------------
# 8. Display Result
# --------------------------------------------------

if prediction[0] == 1:
    print("\n🚨 This email is SPAM")
else:
    print("\n✅ This email is HAM (not spam)")