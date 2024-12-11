import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Create output directory
output_dir = "eda_results"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
true_news_path = "True.csv"
fake_news_path = "Fake.csv"

true_df = pd.read_csv(true_news_path)
fake_df = pd.read_csv(fake_news_path)

# Add labels to datasets
true_df['label'] = 'True'
fake_df['label'] = 'Fake'

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Dataset summary
with open(os.path.join(output_dir, "dataset_summary.txt"), "w") as f:
    f.write("Dataset Overview\n")
    f.write("="*50 + "\n")
    f.write(f"Number of articles: {len(df)}\n")
    f.write(f"Number of missing values:\n{df.isnull().sum()}\n\n")
    f.write("Descriptive Statistics:\n")
    f.write(f"{df.describe(include='all')}\n")

# Class distribution plot
plt.figure(figsize=(8, 6))
df['label'].value_counts().plot(kind='bar', color=['green', 'red'], alpha=0.7)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Articles')
plt.xticks(rotation=0)
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# Text length analysis
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
true_length = df[df['label'] == 'True']['text_length']
fake_length = df[df['label'] == 'Fake']['text_length']

plt.figure(figsize=(10, 6))
true_length.hist(bins=50, alpha=0.7, label='True', color='green')
fake_length.hist(bins=50, alpha=0.7, label='Fake', color='red')
plt.title('Text Length Distribution by Class')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()
plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
plt.close()

with open(os.path.join(output_dir, "text_length_summary.txt"), "w") as f:
    f.write(f"True News Text Length:\n{true_length.describe()}\n\n")
    f.write(f"Fake News Text Length:\n{fake_length.describe()}\n")

# Sentiment analysis
df['sentiment_polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment_subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='sentiment_polarity', hue='label', kde=True, bins=50, palette=['green', 'red'])
plt.title('Sentiment Polarity Distribution by Class')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, "sentiment_distribution.png"))
plt.close()

# Word clouds
true_text = " ".join(str(text) for text in df[df['label'] == 'True']['text'])
fake_text = " ".join(str(text) for text in df[df['label'] == 'Fake']['text'])

WordCloud(width=800, height=400, background_color='white').generate(true_text).to_file(os.path.join(output_dir, "true_news_wordcloud.png"))
WordCloud(width=800, height=400, background_color='white').generate(fake_text).to_file(os.path.join(output_dir, "fake_news_wordcloud.png"))

# N-grams analysis
def plot_ngrams(label, n, num_words=20):
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(n, n), max_features=10000)
    X = vectorizer.fit_transform(df[df['label'] == label]['text'])
    word_counts = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()
    top_indices = word_counts.argsort()[-num_words:][::-1]
    top_words = [(words[i], word_counts[i]) for i in top_indices]

    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.barh(words[::-1], counts[::-1], color='blue')
    plt.xlabel('Frequency')
    plt.title(f'Top {num_words} {n}-grams - {label} News')
    plt.savefig(os.path.join(output_dir, f"top_{n}grams_{label.lower()}.png"))
    plt.close()

plot_ngrams('True', n=2)
plot_ngrams('Fake', n=2)
plot_ngrams('True', n=3)
plot_ngrams('Fake', n=3)
