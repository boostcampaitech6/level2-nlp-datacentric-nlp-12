import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import evaluate


def compute_adjusted_f1(pred, target) :
    f1 = evaluate.load('f1')
    metric = f1.compute(predictions=pred, references=target, average='macro')
    metric["adjusted-f1"] = round((round(metric["f1"],4)+0.972)/2.25, 4)
    return metric


def parse_arguments() :
        
    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_dir', type=str, default='../model/baseline')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--dev_dir', type=str, default="../data/dev.csv")
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

# metric 보여주기
dataset_dev = pd.read_csv(args.dev_dir)
print("Macro F1 Score : {}".format(compute_adjusted_f1(preds, dataset_dev["target"].to_list())))

if args.save :
    dataset_test.to_csv(args.output_dir, index=False)