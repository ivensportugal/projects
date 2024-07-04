# HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

# OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Streamlit
import streamlit as st

model_id = "gpt2"

llm = pipeline(task="text-generation", model=model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)
hf = HuggingFacePipeline(pipeline=pipe)


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

answer = chain.invoke({"student" : student})

st.text(answer)


# agent = create_csv_agent(
#     llm=llm,
#     path="classlist.csv",
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     allow_dangerous_code=True,
# )








