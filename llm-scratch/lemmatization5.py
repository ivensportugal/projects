import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

sentence = 'This is an example of a lemmatized word. The correct is lemma.'

tokens = word_tokenize(sentence)


nltk.download('wordnet')

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()

pos_tags = pos_tag(tokens)

lemmatized_with_pos = [
	lemmatizer.lemmatize(word, pos='v' if tag.startswith('V') else 'n')
	for word, tag in pos_tags
]

print(lemmatized_with_pos)