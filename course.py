# HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

# OpenAI
from langchain_core.prompts import PromptTemplate

# Streamlit
import streamlit as st

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

st.title("CS 846 - SE for BD and AI")
col1, col2, col3 = st.columns(3)
with col1:
	st.text("What is the title of ")
with col2:
	student = st.selectbox("",["A", "B"])
with col3:
	st.text("'s presentation?")

template = "What is the title of {student}'s presentation?"
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

title = chain.invoke({"student" : student})

st.text(title)

