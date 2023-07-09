import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the Netflix dataset
netflix_data = pd.read_csv('netflix_titles.csv', encoding='iso-8859-1')

# Create a new column for sentiment scores of movie and TV show titles
sia = SentimentIntensityAnalyzer()
netflix_data['sentiment_scores'] = netflix_data['Title'].apply(lambda x: sia.polarity_scores(x))

# Extract the compound sentiment score from the sentiment scores dictionary
netflix_data['sentiment_score'] = netflix_data['sentiment_scores'].apply(lambda x: x['compound'])

# Group the data by language and calculate the average sentiment score for movies and TV shows in each language
language_sentiment = netflix_data.groupby('Language')['sentiment_score'].mean()

# Print the top 10 languages with the highest average sentiment score for movies and TV shows
print(language_sentiment.sort_values(ascending=False).head(10))
