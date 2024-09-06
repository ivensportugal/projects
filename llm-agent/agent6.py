from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[], additional_authorized_imports=['requests', 'bs4'])
agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")
