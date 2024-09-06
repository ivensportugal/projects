from transformers import load_tool, CodeAgent

model_download_tool = load_tool('ivensportugal/hf-model-downloads')
agent = CodeAgent(tools=[model_download_tool], llm_engine=llm_engine)

a = agent.run('Can you give me the name of the model that has the most downloads in the "text-to-video" task on the Hugging Face Hub?')
print(a)