import nltk
nltk.download('wordnet')
#nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ['running','better','cats','geese','went']

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized_words)
