# HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

# OpenAI
from langchain_core.prompts import PromptTemplate
# from langchain_core.chains import LLMChain

model_id = "openai-community/gpt2"

llm = pipeline(task="text-generation", model=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)


capital_template = "What is the capital of {country}?"
capital_prompt = PromptTemplate.from_template(capital_template)

chain = capital_prompt | hf

country = "Brazil"

# country_chain = LLMChain(llm=llm, prompt=capital_prompt)
answer = chain.invoke({"country": country})

print(answer)


# prompt.format(country="Brazil")


# pipe = pipeline("text2text-generation", model="google/flan-t5-base")
# print(pipe("What is the capital of Brazil?")['text_generated'])

