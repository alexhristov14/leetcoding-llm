import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import os

data = pd.read_csv("merged.csv")

# Load the saved model and tokenizer
modelpath = "./model-finetuned" if os.path.exists("./model-finetuned") else "gpt2"
tokenizerpath = "./tokenizer-finetuned" if os.path.exists("./tokenizer-finetuned") else "gpt2"
model = GPT2LMHeadModel.from_pretrained(modelpath)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizerpath)

# Ensure the model is in evaluation mode
model.eval()

# Example input prompt
input_prompt = 'Problem:' + data["Question Text"][0] + ' Solution:'

# Tokenize the input prompt
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

# Move input_ids to the appropriate device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_ids = input_ids.to(device)
model.to(device)

# Generate predictions (text continuation)
with torch.no_grad():  # Disable gradient computation for inference
    outputs = model.generate(
        input_ids,
        max_length=200,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True 
    )

# Decode the generated token IDs into text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
