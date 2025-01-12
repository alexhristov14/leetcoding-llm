import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import os

# Load the dataset
data = pd.read_csv("merged.csv")
leetcode_problems = data["Question Text"]
leetcode_solutions = data["Solution_Path"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and tokenizer
modelpath = "./model-finetuned" if os.path.exists("./model-finetuned") else "gpt2"
tokenizerpath = "./tokenizer-finetuned" if os.path.exists("./tokenizer-finetuned") else "gpt2"
model = GPT2LMHeadModel.from_pretrained(modelpath)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizerpath)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

print(f"Model loaded: {modelpath}")

# Dataset definition
class LeetcodeDataset(Dataset):
    def __init__(self, leetcode_problems, leetcode_solutions, tokenizer, max_len):
        self.leetcode_problems = leetcode_problems
        self.leetcode_solutions = leetcode_solutions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.leetcode_problems)

    def __getitem__(self, idx):
        problem = self.leetcode_problems.iloc[idx]
        code = self.leetcode_solutions.iloc[idx]
        
        with open(code, 'r') as file:
            solution_code = file.read()

        input_text = f"Problem: {problem} Solution:"
        input_encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        output_encoding = self.tokenizer(
            solution_code,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        input_ids = input_encoding['input_ids'].squeeze(0)
        attention_mask = input_encoding['attention_mask'].squeeze(0)
        output_ids = output_encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids,
        }

# Dataloader
dataset = LeetcodeDataset(leetcode_problems, leetcode_solutions, tokenizer, max_len=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader)}")


model.save_pretrained("./model-finetuned")
tokenizer.save_pretrained("./tokenizer-finetuned")