
from libs import *
from transformers import pipeline

# Initialize the text generation pipeline with the Qwen model
pipe = pipeline("text-generation", model="Qwen/Qwen2-1.5B")

# Define the system and user prompts
messages = [
    {"role": "system", "content": "You are a highly knowledgeable assistant named Ashley."},
    {"role": "user", "content": "Who are you?"},
]

# Generate a response using the pipeline
response = pipe(messages)

# Print the response
print(response)
