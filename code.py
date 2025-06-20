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

 Step 5: EDA - Sentiment counts
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='textblob_sentiment', palette='pastel')
plt.title('Sentiment Distribution (TextBlob)')
plt.show()

# Step 6: Review length
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(7, 4))
sns.histplot(df['review_length'], bins=50, kde=True, color='skyblue')
plt.title("Review Length Distribution")
plt.show()

# Step 7: Word frequency
def top_words(df, sentiment, n=15):
    text = " ".join(df[df['textblob_sentiment'] == sentiment]['review'])
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return pd.DataFrame(Counter(text.split()).most_common(n), columns=['word', 'count'])

pos_words = top_words(df, 'Positive')
neg_words = top_words(df, 'Negative')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(data=pos_words, y='word', x='count', ax=axes[0], palette='Greens_r')
axes[0].set_title('Top Positive Words')
sns.barplot(data=neg_words, y='word', x='count', ax=axes[1], palette='Reds_r')
axes[1].set_title('Top Negative Words')
plt.tight_layout()
plt.show()

# Step 8: WordCloud
neg_text = " ".join(df[df['textblob_sentiment'] == 'Negative']['review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neg_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud for Negative Reviews")
plt.show()

# Step 9: Sentiment Comparison (TextBlob vs VADER)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df.sample(1000), x='polarity', y='compound', hue='textblob_sentiment', alpha=0.6)
plt.title("TextBlob vs VADER Compound Sentiment")
plt.show()

# Step 10: Violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(data=df.sample(2000), x='textblob_sentiment', y='compound', palette='pastel')
plt.title("VADER Compound by TextBlob Sentiment")
plt.show()

# Step 11: Export results
df.to_csv("final_sentiment_results.csv", index=False)
print("✔️ Sentiment analysis complete. Results saved to final_sentiment_results.csv")

