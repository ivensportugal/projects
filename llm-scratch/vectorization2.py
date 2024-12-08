import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

import math

nltk.download('punkt')

# corpus = [
# 	'I love machine learning',
# 	'Word2Vec is a powerful model',
# 	'Natural language processing is fun',
# 	'Deep learning is a subfield of machine learning'
# ]

corpus = [
	'You should study math',
	'He should study math',
	'Study math and you will like it',
	'They who study math will succeed',
	'A reminder to study math',
	'Do not forget to study math'
]

tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
# print('tokenized_corpus')
# print(tokenized_corpus)
# print('end of tokenized_corpus')

model = Word2Vec(sentences=tokenized_corpus, vector_size=3, window=1, min_count=1, sg=0)

# words = set()
# for sentence in corpus:
# 	words.update(sentence.split())

# for word in words:
# 	print(word.lower())
# 	print(model.wv[word.lower()])

words = model.wv.index_to_key

for word in words:
	print(word)
	print(model.wv[word])


print('We now calculate the Euclidean distance of all vectors')
print('in an attempt to find semantic relationships')

distance = {}

for word1 in words:
	for word2 in words:
		vector1 = model.wv[word1]
		vector2 = model.wv[word2]

		squared_diff = sum((x-y) ** 2 for x, y in zip(vector1, vector2))
		distance.update({(word1, word2): math.sqrt(squared_diff)})

# print('distance')
# for i in distance:
# 	print(i, distance[i])


print('sorted euclidean distances')
distance_sorted = {k: v for k, v in sorted(distance.items(), key=lambda item: item[1], reverse=True)}
for i in distance_sorted:
	print(i, distance_sorted[i])

print('The end')