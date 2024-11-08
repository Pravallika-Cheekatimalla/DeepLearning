#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from collections import Counter
import string
import re
import os

# Function to load and parse the SQuAD dataset into contexts, questions, and answers
def load_squad(path):
    with open(path, 'r') as file:
        data = json.load(file)
        
    contexts, questions, answers = [], [], []
    
    # Extract contexts, questions, and answers from SQuAD data
    for group in data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                answer_key = 'plausible_answers' if 'plausible_answers' in qa else 'answers'
                for answer in qa[answer_key]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

# Load training and validation sets
train_contexts, train_questions, train_answers = load_squad('./Input/spoken_train-v1.1.json')
val_contexts, val_questions, val_answers = load_squad('./Input/spoken_test-v1.1.json')

# Function to adjust answer end indices
def add_end_index(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)
        
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        else:
            for n in range(1, 3):
                if context[start_idx - n:end_idx - n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n

# Apply function to training and validation sets
add_end_index(train_answers, train_contexts)
add_end_index(val_answers, val_contexts)

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize datasets
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

# Function to add token start and end positions to encodings
def set_answer_positions(encodings, answers):
    start_positions, end_positions = [], []
    for i, answer in enumerate(answers):
        start_pos = encodings.char_to_token(i, answer['answer_start'])
        end_pos = encodings.char_to_token(i, answer['answer_end'])
        
        # Adjust if answer is truncated
        if start_pos is None:
            start_pos = tokenizer.model_max_length
        shift = 1
        while end_pos is None:
            end_pos = encodings.char_to_token(i, answer['answer_end'] - shift)
            shift += 1
            
        start_positions.append(start_pos)
        end_positions.append(end_pos)
    
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# Apply function to train and validation encodings
set_answer_positions(train_encodings, train_answers)
set_answer_positions(val_encodings, val_answers)

# Custom Dataset class for SQuAD data
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Build datasets
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

# Initialize model and optimizer
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
optim = AdamW(model.parameters(), lr=2e-6)

# Setup accelerator
accelerator = Accelerator()
model, optim, train_loader = accelerator.prepare(model, optim, DataLoader(train_dataset, batch_size=16, shuffle=True))

# Scheduler setup for linear learning rate decay
num_training_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
for epoch in range(3):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(accelerator.device)
        attention_mask = batch['attention_mask'].to(accelerator.device)
        start_positions = batch['start_positions'].to(accelerator.device)
        end_positions = batch['end_positions'].to(accelerator.device)
        
        outputs = model(input_ids, attention_mask=attention_mask,
                        start_positions=start_positions, end_positions=end_positions)
        
        loss = outputs[0]
        accelerator.backward(loss)
        optim.step()
        scheduler.step()
        
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item(), lr=optim.param_groups[0]['lr'])

# Save model and tokenizer
save_path = 'models/distilbert-improved'
if not os.path.exists(save_path):
    os.makedirs(save_path)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Evaluation
val_loader = DataLoader(val_dataset, batch_size=16)

answers, references = [], []
model.eval()

with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(accelerator.device)
        attention_mask = batch['attention_mask'].to(accelerator.device)
        start_true = batch['start_positions'].to(accelerator.device)
        end_true = batch['end_positions'].to(accelerator.device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        
        for i in range(start_pred.shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
            pred_ans = ' '.join(tokens[start_pred[i]: end_pred[i] + 1])
            true_ans = ' '.join(tokens[start_true[i]: end_true[i] + 1])
            answers.append(pred_ans)
            references.append(true_ans)

# Helper functions for EM and F1 Score calculations
def normalize_answer(text):
    def remove_articles(txt):
        return re.sub(r'\b(a|an|the)\b', ' ', txt)

    def white_space_fix(txt):
        return ' '.join(txt.split())

    def remove_punctuation(txt):
        exclude = set(string.punctuation)
        return ''.join(char for char in txt if char not in exclude)

    return white_space_fix(remove_articles(remove_punctuation(text.lower())))

def exact_match(pred, truth):
    return normalize_answer(pred) == normalize_answer(truth)

def f1_score(pred, truth):
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

# Evaluate function to calculate EM and F1 scores
def evaluate(answers, references):
    em, f1 = 0, 0
    for pred, ref in zip(answers, references):
        em += exact_match(pred, ref)
        f1 += f1_score(pred, ref)
    
    total = len(answers)
    return {'Exact Match': 100 * em / total, 'F1 Score': 100 * f1 / total}

# Calculate and print the evaluation metrics
metrics = evaluate(answers, references)
print(f"Exact Match: {metrics['Exact Match']:.2f}")
print(f"F1 Score: {metrics['F1 Score']:.2f}")


# In[ ]:




