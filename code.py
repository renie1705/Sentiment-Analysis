import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import requests
import tarfile
import os
import string
from collections import Counter

# Downloads
nltk.download("vader_lexicon")

# Step 1: Download IMDb dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
response = requests.get(url)
with open("aclImdb_v1.tar.gz", "wb") as f:
    f.write(response.content)
with tarfile.open("aclImdb_v1.tar.gz", "r:gz") as tar:
    tar.extractall()

# Step 2: Load dataset
def load_reviews(folder, sentiment):
    reviews = []
    path = f"aclImdb/train/{sentiment}"
    for file in os.listdir(path):
        with open(os.path.join(path, file), encoding="utf-8") as f:
            reviews.append([f.read(), sentiment])
    return reviews

pos_reviews = load_reviews("aclImdb/train", "pos")
neg_reviews = load_reviews("aclImdb/train", "neg")
df = pd.DataFrame(pos_reviews + neg_reviews, columns=["review", "label"])
df = df.sample(frac=1).reset_index(drop=True)

# Step 3: Add TextBlob sentiment
df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['textblob_sentiment'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Step 4: Add VADER sentiment
sid = SentimentIntensityAnalyzer()
df['vader'] = df['review'].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['vader'], axis=1), df['vader'].apply(pd.Series)], axis=1)
