import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("social_media_reviews.csv")

# IMPORTANT: dataset must have these columns
# Column 0 → text
# Column 1 → sentiment (positive, negative, neutral)

text_col = data.columns[0]
sentiment_col = data.columns[1]

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

data['clean_text'] = data[text_col].apply(clean_text)

X = data['clean_text']
y = data[sentiment_col].astype(str).str.lower()

vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)

def predict_sentiment(text):
    cleaned = clean_text(text)

    # RULE BASED NEUTRAL (FOR DEMO + EXAM)
    neutral_words = ["okay", "average", "fine", "normal", "not bad", "decent"]
    if any(word in cleaned for word in neutral_words):
        return "Neutral"

    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]

    if pred == "positive":
        return "Positive"
    else:
        return "Negative"
