from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)

agent.run("Why does Mike not know many people in New York?", audio="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3")

print(agent.system_prompt_template)