# Generic
from dotenv import load_dotenv

# HuggingFace
from transformers import pipeline

# # OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

pipe = pipeline(task="text-generation", model="openai-community/gpt2")
answer = pipe('What is the capital of Brazil?', pad_token_id=pipe.tokenizer.eos_token_id)
print(answer)
print('***')
print(answer[0]['generated_text'])
print('***')



# prompt = PromptTemplate(
# 	input_variables = ['country'],
# 	template = "What is the capital of {country}?"
# 	)

# LLMChain(llm = )

# prompt.format(country="Brazil")


# pipe = pipeline("text2text-generation", model="google/flan-t5-base")
# print(pipe("What is the capital of Brazil?")['text_generated'])

