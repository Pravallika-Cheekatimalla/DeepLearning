#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import torch
from collections import Counter
import string
import re
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import os

# Function to load the SQuAD dataset and parse it into contexts, questions, and answers
def load_squad_data(path):
    with open(path, 'r') as f:
        data = json.load(f)

    contexts, questions, answers = [], [], []

    for entry in data['data']:
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers_field = 'plausible_answers' if 'plausible_answers' in qa else 'answers'
                for answer in qa[answers_field]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    return contexts, questions, answers

# Load training and validation data
train_contexts, train_questions, train_answers = load_squad_data('./Input/spoken_train-v1.1.json')
val_contexts, val_questions, val_answers = load_squad_data('./Input/spoken_test-v1.1.json')

# Function to add the end index to answers
def adjust_end_index(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n

# Adjust end indices in training and validation sets
adjust_end_index(train_answers, train_contexts)
adjust_end_index(val_answers, val_contexts)

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize training and validation sets
train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

# Function to add token positions for answers
def add_answer_positions(encodings, answers):
    start_positions, end_positions = [], []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# Apply token positions function
add_answer_positions(train_encodings, train_answers)
add_answer_positions(val_encodings, val_answers)

# Create a custom Dataset class for PyTorch
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Prepare datasets for training and validation
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

# Initialize the model
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-6)

# DataLoader for training
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop
for epoch in range(3):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        
        # Extract batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask,
                         start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

# Save the model
model_path = 'models/distilbert-base-model'
if not os.path.exists(model_path):
    os.makedirs(model_path)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Load the model for evaluation
model = DistilBertForQuestionAnswering.from_pretrained(model_path)
model.to(device)

# Set model to evaluation mode
model.eval()

# DataLoader for validation
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize lists for predictions and references
answers, references = [], []
accuracy = []

# Evaluate the model
with torch.no_grad():
    loop = tqdm(val_loader)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)
        
        # Make predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        
        # Calculate accuracy
        accuracy.append(((start_pred == start_true).sum() / len(start_pred)).item())
        accuracy.append(((end_pred == end_true).sum() / len(end_pred)).item())
        
        for i in range(start_pred.shape[0]):
            all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
            pred_answer = ' '.join(all_tokens[start_pred[i]: end_pred[i] + 1])
            true_answer = ' '.join(all_tokens[start_true[i]: end_true[i] + 1])
            answers.append(pred_answer)
            references.append(true_answer)

# Function to normalize answers for EM and F1 calculation
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

# Exact Match and F1 Score calculation
def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    true_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Evaluation function
def evaluate(predictions, references):
    exact_match, f1 = 0, 0
    for pred, ref in zip(predictions, references):
        exact_match += exact_match_score(pred, ref)
        f1 += f1_score(pred, ref)
    
    return {'exact_match': 100.0 * exact_match / len(predictions), 'f1': 100.0 * f1 / len(predictions)}

# Calculate Exact Match and F1 score
metrics = evaluate(answers, references)
print(f"Exact Match: {metrics['exact_match']:.2f}")
print(f"F1 Score: {metrics['f1']:.2f}")


# In[ ]:




