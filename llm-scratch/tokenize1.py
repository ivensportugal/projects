import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize, sent_tokenize

sentence = "Hello world. How are you? You are good."

words = word_tokenize(sentence)
print(words)

sentences = sent_tokenize(sentence)
print(sentences)
