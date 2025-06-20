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
