
import re
from pathlib import Path
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download language resources when needed
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

PROJECT_DIR = Path("/content/sentiment_project")

model = joblib.load(PROJECT_DIR / "sentiment_model.pkl")
tfidf = joblib.load(PROJECT_DIR / "tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]
    return " ".join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])
    prediction = model.predict(features)[0]

    labels = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    return labels[prediction]

if __name__ == "_main_":
    examples = [
        "I absolutely love this product!",
        "This is the worst experience ever.",
        "The product arrived today."
    ]

    for sentence in examples:
        print(f"Text: {sentence}")
        print(f"Sentiment: {predict_sentiment(sentence)}")
        print()
