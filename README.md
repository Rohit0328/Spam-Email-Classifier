#  Spam Email Classifier

A machine learning based spam email detection system that classifies emails as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and machine learning.

The project uses the **SpamAssassin dataset**, TF-IDF for text feature extraction, and multiple machine learning algorithms to compare their performance.

---

##  Project Overview

Spam emails are unwanted messages that can contain advertisements, scams, misleading information, or potentially harmful content.

The goal of this project is to build a machine learning system that learns patterns from previously labeled emails and uses those patterns to classify new emails as:

-  Spam
-  Ham (Legitimate)

The project follows a complete machine learning pipeline from text preprocessing and feature extraction to model training, evaluation, and prediction.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Joblib

---

##  Methodology

The project follows this pipeline:

```text
Raw Emails
     ↓
Data Preparation
     ↓
Text Preprocessing
     ↓
Train-Test Split
     ↓
TF-IDF Vectorization
     ↓
Model Training
     ↓
Model Comparison
     ↓
Final Linear SVM
     ↓
Spam / Ham Prediction

## Model Performance

The models were evaluated on **1,870 unseen test emails** using accuracy, precision, recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.27% | 97% | 92% | 95% |
| Multinomial Naive Bayes | 86.90% | 98% | 50% | 66% |
| Linear SVM | 99.63% | 100% | 99% | 99% |

> Precision, recall, and F1-score values are rounded values from the classification reports.

### Final Model: Linear SVM

The Linear SVM achieved an accuracy of 99.63% on the test set.

### Confusion Matrix

```text
                    Predicted
                 Ham       Spam

Actual Ham       1389        1
Actual Spam         6      474
