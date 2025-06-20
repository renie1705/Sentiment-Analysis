plt.figure(figsize=(8, 5))
sns.violinplot(data=df.sample(2000), x='textblob_sentiment', y='compound', palette='pastel')
plt.title("VADER Compound by TextBlob Sentiment")
plt.show()
