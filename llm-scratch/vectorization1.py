import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# nltk.download('punkt')

corpus = [
	'I love machine learning.',
	'I learned machine learning.',
	'When I learn machine learning, I am happy.',
	'Yesterday, I learned machine learning. I learn very fast.',
	'Word2Vec is a powerful model.',
	'Natural language processing is fun.',
	'Deep learning is a subfield of machine learning.'
]

tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

print('tokenized corpus')
print(tokenized_corpus)

model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1,sg=0)

print('model')
print(model.wv['machine'])