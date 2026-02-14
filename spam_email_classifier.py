# Spam Email Classification Project
# Author: Manisha Juttu

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "email": [
        "Win money now",
        "Limited time offer",
        "Meeting scheduled tomorrow",
        "Project discussion update",
        "Congratulations you won lottery",
        "Let's complete the assignment"
    ],
    "label": [1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["email"], df["label"], test_size=0.3, random_state=42
)

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict
predictions = model.predict(X_test_vec)

# Check accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Test custom email
test_email = ["Congratulations! You have won a free ticket"]
test_vec = vectorizer.transform(test_email)
result = model.predict(test_vec)

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")
