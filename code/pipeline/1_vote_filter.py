import pandas as pd
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from collections import Counter



# 만장일치인지 확인한 후, 맞으면 만장일치된 값을 final에 리턴. 아닐 경우 -1을 리턴.
def check_same(row):
  values = [row['target_yobi'], row['target_JiHoon-kim'], row['target_jihoonkimharu'], row["target_chunwoolee0"], row['target_shangrilar']]
  counts = Counter(values)
  if len(counts) == 1 :
    return values[0]
  else:
    return -1
  

def parse_arguments() :

  parser = argparse.ArgumentParser(description='Argparse')

  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--target_dir', type=str, default='../data/train.csv')
  parser.add_argument('--output_dir', type=str, default='../data/relabel.csv')
  
  args = parser.parse_args()

  return args


args = parse_arguments()

df_org = pd.read_csv(args.target_dir)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE

# Batch the data
batches = [df_org[i:i+args.batch_size] for i in range(0, len(df_org), args.batch_size)]
model_names = [
    "yobi/klue-roberta-base-ynat",
    "JiHoon-kim/bert-base-klue-ynat-finetuned",
    "jihoonkimharu/bert-base-klue-ynat-finetuned",
    "chunwoolee0/klue_ynat_roberta_base_model",
    "shangrilar/roberta-base-klue-ynat-classification",
]

for model_name in model_names :

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
    model.eval()

    # Perform inference on each batch
    preds = []
    for batch in tqdm(batches):
        inputs = tokenizer(batch['text'].tolist(), return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    # Update the test data with the predictions
    df_org['target_{}'.format(model_name.split("/")[0])] = preds



# 만장일치된 값으로 target 수정. (target drop하고 target_final을 target으로 rename해줌.)
df_org['target_final'] = df_org.apply(check_same, axis=1)

# 만장일치 된것만 남기기
df_sure = df_org[df_org['target_final'] != -1]
df_sure.drop('target', axis=1, inplace=True)
df_sure.rename(columns={'target_final': 'target'}, inplace=True)

# 저장
df_sure.to_csv(args.output_dir, index=False)
print("Done!")