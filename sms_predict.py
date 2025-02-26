from flask import Flask, request, render_template
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)

df = pd.read_csv("spam_data.txt", sep="|", names=["label", "message"], skiprows=1)

df["label"] = df["label"].str.strip().str.lower()
df["message"] = df["message"].str.strip()

df["label"] = df["label"].map({"ham": 0, "spam": 1})



lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text=text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)

    words = word_tokenize(text)

    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    return " ".join(words)

df["cleaned_message"] = df["message"].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["cleaned_message"])

y = df["label"].values

# print ("Shape of TF-IDF matrix: ", X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
### Naive Bayes Classifier
# nb_classifier = MultinomialNB()
# nb_classifier.fit(X_train, y_train)

# y_pred_nb = nb_classifier.predict(X_test)

# accuracy_nb = accuracy_score(y_test, y_pred_nb)

# print (f"Naive Bayes Accuracy: {accuracy_nb:.2f}")
# print (classification_report(y_test, y_pred_nb))
# print ("----------------------------------------------------")

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

y_pred_lr = lr_classifier.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
# print (f"Logistic Regression Accuracy: {accuracy_lr:.2f}")
# print (classification_report(y_test, y_pred_lr))


def predict_spam(message):
    message = preprocess_text(message)
    vectorized_message = vectorizer.transform([message])
    prediction = lr_classifier.predict(vectorized_message)[0]
    return "Spam" if prediction == 1 else "Not Spam"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        message = request.form.get("message")
        if message:
            result = predict_spam(message)
    LRAccuracy = round(accuracy_lr, 2)
    #report = classification_report(y_test, y_pred_lr)

    return render_template("index.html", result=result, LRAccuracy = LRAccuracy)


if __name__ == '__main__':
    app.run(debug=True)
