# Spam Email Detector using Machine Learning (NLP)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# STEP 1: Create Sample Dataset
# -----------------------------
data = {
    "email": [
        "Win money now!!! Click here",
        "Hi, how are you doing today",
        "Congratulations you won a lottery",
        "Let's meet for lunch tomorrow",
        "Free entry in a prize contest",
        "Project meeting scheduled at 10am",
        "Claim your free gift card now",
        "Can you send me the notes?"
    ],
    "label": [
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# -----------------------------
# STEP 2: Features & Labels
# -----------------------------
X = df["email"]
y = df["label"]

# -----------------------------
# STEP 3: Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------------
# STEP 4: Text Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# STEP 5: Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# STEP 6: Predictions
# -----------------------------
predictions = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nReport:\n", classification_report(y_test, predictions))

# -----------------------------
# STEP 7: Test Custom Email
# -----------------------------
new_email = ["Congratulations! You have won free tickets"]
new_email_vec = vectorizer.transform(new_email)

result = model.predict(new_email_vec)

print("\nNew Email Prediction:", result[0])
