from huggingface_hub import InferenceClient

# Initialize the Inference Client with the model you want to use
client = InferenceClient(model="gpt-3.5-turbo")

# Define the input data
input_data = "What is the capital of France?"

# Make a prediction
response = client.predict(input_data)

# Print the response
print(response)
