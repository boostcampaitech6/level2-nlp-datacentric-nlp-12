import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

import torch
from torch.utils.data import Dataset

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from sklearn.model_selection import train_test_split


class BERTDataset(Dataset):
    def __init__(self, data, tokenizer):
        input_texts = data['text']
        targets = data['target']
        self.inputs = []
        self.labels = []
        
        for text, label in zip(input_texts, targets):
            tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
            self.inputs.append(tokenized_input)
            self.labels.append(torch.tensor(label))
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),  
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }
    
    def __len__(self):
        return len(self.labels)
    


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')


def parse_arguments() :
        
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--train_dir', type=str, default="../data/train.csv")
    parser.add_argument('--dev_dir', type=str, default="../data/dev.csv")
    parser.add_argument('--save_total_limit', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--dev_ratio', type=float, default=0.3)
    args = parser.parse_args()

    return args



args = parse_arguments()


SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

    

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE



BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')


model_name = 'klue/bert-base'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset_train = pd.read_csv(args.train_dir)
dataset_valid = pd.read_csv(args.dev_dir)
# dataset_train, dataset_valid = train_test_split(data, test_size=args.dev_ratio, stratify=data['target'],random_state=SEED)

data_train = BERTDataset(dataset_train, tokenizer)
data_valid = BERTDataset(dataset_valid, tokenizer)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

f1 = evaluate.load('f1')



training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    do_predict=True,
    logging_strategy='steps',
    evaluation_strategy='steps',
    save_strategy='steps',
    logging_steps=100,
    eval_steps=args.eval_steps,
    save_steps=args.eval_steps,
    save_total_limit=args.save_total_limit,
    learning_rate= 2e-05,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon=1e-08,
    weight_decay=0.01,
    lr_scheduler_type='linear',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_f1',
    greater_is_better=True,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_train,
    eval_dataset=data_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained('../best_model')