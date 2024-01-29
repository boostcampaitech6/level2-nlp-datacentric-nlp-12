import pandas as pd
import torch
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


INPUT_PATH = '/content/drive/MyDrive/Level2_Topic/'
gemini_gen = pd.read_csv(f'{INPUT_PATH}gemini_generate_V2.csv')

# Batch the data
batch_size = 128
gemini_batches = [gemini_gen[i:i+batch_size] for i in range(0, len(gemini_gen), batch_size)]

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE

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
    for batch in tqdm(gemini_batches):
        inputs = tokenizer(batch['text'].tolist(), return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    # Update the test data with the predictions
    gemini_gen['target_{}'.format(model_name.split("/")[0])] = preds



# 만장일치된 값으로 target 수정. (target drop하고 target_final을 target으로 rename해줌.)
gemini_gen['target_final'] = gemini_gen.apply(check_same, axis=1)

# 만장일치 된것만 남기기
gemini_sure = gemini_gen[gemini_gen['target_final'] != -1]
gemini_sure.drop('target', axis=1, inplace=True)
gemini_sure.rename(columns={'target_final': 'target'}, inplace=True)

# 저장
gemini_sure.to_csv(f'{INPUT_PATH}gemini_relabel.csv', index=False)