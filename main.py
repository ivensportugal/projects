# HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

# OpenAI
from langchain_core.prompts import PromptTemplate

model_id = "gpt2"

llm = pipeline(task="text-generation", model=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)


capital_template = "What is the capital of {country}?"
capital_prompt = PromptTemplate.from_template(capital_template)

population_template = "What is the population of {capital}?"
population_prompt = PromptTemplate.from_template(population_template)

chain = capital_prompt | population_prompt | hf

country = "Brazil"

answer = chain.invoke({"country": country})

print(type(answer))
print(answer)