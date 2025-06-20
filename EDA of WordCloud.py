neg_text = " ".join(df[df['textblob_sentiment'] == 'Negative']['review'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neg_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud for Negative Reviews")
plt.show(
