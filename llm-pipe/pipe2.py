from transformers import pipeline

def data():
  for i in ['Brazil','Canada','USA']:
    yield f'What is the capital of {i}? (short answer)'

pipe = pipeline(model='openai-community/gpt2', device=0)
generated_characters = 0
for out in pipe(data()):
  generated_characters += len(out[0]['generated_text'])
  print(out[0]['generated_text'])
