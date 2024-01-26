import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_arguments() :
        
    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default='../best_model')
    parser.add_argument('--output_dir', type=str, default='../output.csv')
    
    args = parser.parse_args()

    return args


args = parse_arguments()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')

model_name = 'klue/bert-base'


# test dataset batchify
dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
batch_size = args.batch_size
test_batches = [dataset_test[i:i+batch_size] for i in range(0, len(dataset_test), batch_size)]

# tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=7).to(DEVICE)
model.eval()

# Perform inference on each batch
preds = []
for batch in tqdm(test_batches):
    inputs = tokenizer(batch['text'].tolist(), return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
        preds.extend(pred)

dataset_test['target'] = preds
dataset_test.to_csv(args.output_dir, index=False)