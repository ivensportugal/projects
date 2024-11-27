import spacy
import en_core_web_sm

nlp = spacy.load('en_core_web_sm')

text = 'This is a long text. I am happy.'

doc = nlp(text)

print('Noun phrases:', [chunk.text for chunk in doc.noun_chunks])
print('Verbs:', [token.lemma_ for token in doc if token.pos_ == 'VERB'])

print('------------')
print('Noun phrases')
print(doc.noun_chunks)
print('')

print('------------')
print('Verbs')
print(doc)


print('------------')
print('elements in doc')



for entity in doc.ents:
	print(entity.text, entity.label_)