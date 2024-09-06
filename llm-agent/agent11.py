from langchain.agents import load_tools
from transformers import Tool, ReactCodeAgent

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = ReactCodeAgent(tools=[search_tool])

agent.run("How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?")