from transformers import pipeline

classifier = pipeline(model='facebook/bart-large-mnli')
answer = classifier('I have a problem with my iphone that needs to be resolved asap!!', candidate_labels=['urgent', 'not urgent', 'phone', 'tablet', 'computer'])
print(answer)