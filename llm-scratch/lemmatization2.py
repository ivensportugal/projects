import nltk
nltk.download('wordnet')
#nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

lemmatizer = WordNetLemmatizer()
lemmatized_words_pos = [
  lemmatizer.lemmatize('running', pos='v'),
  lemmatizer.lemmatize('better', pos='a'),
  lemmatizer.lemmatize('cats', pos='n')
]

print(lemmatized_words_pos)

