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

    parser.add_argument('--model_dir', type=str, default='../best_model')
    parser.add_argument('--output_dir', type=str, default='../output.csv')
    
    args = parser.parse_args()

    return args


args = parse_arguments()

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

model_name = 'klue/bert-base'

# 모델 불러오기만 여기서 해주세요.
dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, num_labels=7).to(DEVICE)

model.eval()
preds = []
for idx, sample in tqdm(dataset_test.iterrows()):
    inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
        preds.extend(pred)
        
dataset_test['target'] = preds
dataset_test.to_csv(args.output_dir, index=False)