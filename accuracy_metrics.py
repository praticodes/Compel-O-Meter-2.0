import nltk
import read_csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
ai_lexicon = read_csv.ai_lexicon_words_dict('data/ai_lexicon.csv')
abs_errors = []
py_scores = []
program_scores = []


for word in ai_lexicon:
    py_sentiment_score = sia.polarity_scores(word)['compound']
    program_sentiment_score = ai_lexicon[word]/2
    abs_error = abs(program_sentiment_score-py_sentiment_score)
    abs_errors.append(abs_error)
    py_scores.append(py_sentiment_score)
    program_scores.append(program_sentiment_score)

MAE = sum(abs_errors)/len(abs_errors)
print(py_scores)
print(program_scores)
print(abs_errors)
print(MAE)
