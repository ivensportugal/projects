from huggingface_hub import list_models

task = 'text_classification'

model = next(iter(list_models(filter=task, sort='downloads', direction=-1)))
print(model.id)