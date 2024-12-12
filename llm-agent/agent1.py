from huggingface_hub import login, InferenceClient

login('')

client = InferenceClient(model='meta-llama/Meta-Llama-3-70B-Instruct')

def llm_engine(messages, stop_sequences=['Task']) -> str:
	response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
	answer = response.choices[0].message.content
	return answer

print(llm_engine('hi'))