import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('stopwords')

text = 'Hello world. I am I. You are you. He is he. She is she.'

tokens = word_tokenize(text)

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in punctuation]

vocabulary = set(filtered_tokens)

print('Vocabulary:', vocabulary)