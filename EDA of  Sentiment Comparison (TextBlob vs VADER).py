plt.figure(figsize=(7, 5))
sns.scatterplot(data=df.sample(1000), x='polarity', y='compound', hue='textblob_sentiment', alpha=0.6)
plt.title("TextBlob vs VADER Compound Sentiment")
plt.show()
