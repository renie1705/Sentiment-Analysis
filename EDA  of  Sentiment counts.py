plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='textblob_sentiment', palette='pastel')
plt.title('Sentiment Distribution (TextBlob)')
plt.show()
