import pandas as pd
from tqdm import tqdm
from roundtrip import backtranslate
import time

#use roundtrip library to backtranlate
def roundtrip(sentence: str) -> str:
    translated = backtranslate(phrase=sentence, from_language='ko')
    return translated

translated_sentences = []

def translate(train_path: str) -> pd:
    data = pd.read_csv(train_path, encoding='utf-8')
    origin = data['text']
    for sen in tqdm(origin):
        time.sleep(1)  # HTTPError: HTTP Error 429: Too Many Requests 방지
        translated_sentences.append(roundtrip(sen)) #roundtrip 방식으로 변역
    data['text'] = translated_sentences
    return data


if __name__ == '__main__':
    train_path = "../data/train_clean.csv"
    save_path = f"../data/translated.csv"
    translated_data = translate(train_path=train_path)
    translated_data.to_csv(save_path, encoding='utf-8', index=False)
