import tkinter as tk
from tkinter import messagebox

import re
import string
import joblib
import nltk

from nltk.corpus import stopwords


# --------------------------------------------------
# 1. NLTK Setup
# --------------------------------------------------

nltk.download("stopwords", quiet=True)

stop_words = set(
    stopwords.words("english")
)


# --------------------------------------------------
# 2. Load Model and Vectorizer
# --------------------------------------------------

model = joblib.load(
    "spam_svm_model.pkl"
)

vectorizer = joblib.load(
    "tfidf_vectorizer.pkl"
)

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")


# --------------------------------------------------
# 3. Text Cleaning
# --------------------------------------------------

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Replace email addresses
    text = re.sub(
        r"\S+@\S+",
        " EMAILADDRESS ",
        text
    )

    # Replace URLs
    text = re.sub(
        r"http\S+|www\S+",
        " URL ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Convert tokens back to text
    return " ".join(tokens)


# --------------------------------------------------
# 4. Detect Spam Function
# --------------------------------------------------

def detect_spam():

    print("Detect Spam button clicked!")

    email = email_box.get("1.0", tk.END).strip()

    print("Email entered:", email)

    if not email:
        messagebox.showwarning(
            "Empty Email",
            "Please enter an email first."
        )
        return

    try:

        cleaned_email = clean_text(email)

        print("\nCleaned email:")
        print(cleaned_email)

        email_tfidf = vectorizer.transform(
            [cleaned_email]
        )

        print("\nTF-IDF conversion completed")
        print("Shape:", email_tfidf.shape)

        prediction = model.predict(email_tfidf)

        print("Prediction:", prediction)

        if prediction[0] == 1:

            result_label.config(
                text="SPAM",
                fg="red"
            )

            messagebox.showinfo(
                "Spam Detection Result",
                "🚨 This email is SPAM"
            )

        else:

            result_label.config(
                text="HAM (NOT SPAM)",
                fg="green"
            )

            messagebox.showinfo(
                "Spam Detection Result",
                "✅ This email is HAM (NOT SPAM)"
            )

        root.update()

    except Exception as e:

        print("ERROR:", e)

        messagebox.showerror(
            "Prediction Error",
            str(e)
        )

# --------------------------------------------------
# 5. Clear Function
# --------------------------------------------------

def clear_email():

    email_box.delete(
        "1.0",
        tk.END
    )

    result_label.config(
        text="Result will appear here",
        fg="black"
    )


# --------------------------------------------------
# 6. Create Main Window
# --------------------------------------------------

root = tk.Tk()

root.title(
    "Spam Email Detector"
)

# Increased height so result is visible
root.geometry(
    "700x650"
)


# --------------------------------------------------
# 7. Title
# --------------------------------------------------

title_label = tk.Label(
    root,
    text="📧 Spam Email Detector",
    font=("Arial", 24, "bold")
)

title_label.pack(
    pady=20
)


# --------------------------------------------------
# 8. Instruction
# --------------------------------------------------

instruction_label = tk.Label(
    root,
    text="Paste your email below:",
    font=("Arial", 14)
)

instruction_label.pack(
    pady=10
)


# --------------------------------------------------
# 9. Email Text Box
# --------------------------------------------------

email_box = tk.Text(
    root,
    height=12,
    width=70,
    font=("Arial", 12)
)

email_box.pack(
    padx=20,
    pady=10
)


# --------------------------------------------------
# 10. Detect Button
# --------------------------------------------------

detect_button = tk.Button(
    root,
    text="Detect Spam",
    font=("Arial", 14, "bold"),
    command=detect_spam
)

detect_button.pack(
    pady=10
)


# --------------------------------------------------
# 11. Clear Button
# --------------------------------------------------

clear_button = tk.Button(
    root,
    text="Clear",
    font=("Arial", 12),
    command=clear_email
)

clear_button.pack(
    pady=5
)


# --------------------------------------------------
# 12. Result Label
# --------------------------------------------------

result_label = tk.Label(
    root,
    text="Result will appear here",
    font=("Arial", 20, "bold")
)

result_label.pack(
    pady=20
)


# --------------------------------------------------
# 13. Start Application
# --------------------------------------------------

print("GUI started")

root.mainloop()