# ===== IMPORT LIBRARIES =====
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# ===== LOAD DATASET =====
data = pd.read_csv("social_media_reviews.csv")
print("Dataset Loaded Successfully")
print(data.head())

# ===== CLEAN TEXT =====
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# detect columns automatically
text_col = data.columns[0]
sentiment_col = data.columns[1]

data['clean_text'] = data[text_col].apply(clean_text)

# ===== FEATURE EXTRACTION =====
X = data['clean_text']
y = data[sentiment_col]

vectorizer = TfidfVectorizer(max_features=3000)
X_tfidf = vectorizer.fit_transform(X)

# ===== SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# ===== TRAIN MODEL =====
model = MultinomialNB()
model.fit(X_train, y_train)

# ===== PREDICT =====
y_pred = model.predict(X_test)

# ===== EVALUATE =====
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

# ===== SENTIMENT DISTRIBUTION =====
data[sentiment_col].value_counts().plot(kind='bar')
plt.title("Public Opinion Distribution")
plt.show()

# ===== TEST NEW INPUT =====
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    return model.predict(vect)[0]

print("\nTest Review:")
print("Predicted Sentiment:",
      predict_sentiment("This product is amazing and useful"))
