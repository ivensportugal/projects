import nltk
nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger_eng')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from nltk import pos_tag
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

# Example sentence
sentence = "The cats are running faster than the dogs."
sentence2 = "Hello. How are you? You are good and not good. The end."
sentence3 = "I don't like chocolate-based cookies."
sentence4 = "Salut. J'ai ici."
sentence5 = "Today, I am extremely happy because the laptop has arrived."

# Tokenize and POS tag the sentence
tokens = word_tokenize(sentence5)
pos_tags = pos_tag(tokens)

# Lemmatize based on POS tags
lemmatized_with_pos = [
    lemmatizer.lemmatize(word, pos='v' if tag.startswith('V') else 'n') 
    for word, tag in pos_tags
]

print(lemmatized_with_pos)
# Output: ['The', 'cat', 'are', 'run', 'faster', 'than', 'the', 'dog', '.']
