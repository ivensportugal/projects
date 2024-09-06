from transformers import CodeAgent, HfEngine

# llm_engine = HfEngine(model='meta-llama/Meta-Llama-3-70B-Instruct')
# agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)

agent = CodeAgent(tools=[], add_base_tools=True)

agent.run('Could you translate this sentence from French, say it out loud and return the audio?',
	sentence='Où est la boulangerie la plus proche?',)