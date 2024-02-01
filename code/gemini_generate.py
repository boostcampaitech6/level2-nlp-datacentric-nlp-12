import textwrap
import argparse

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

import pandas as pd

import os
from dotenv import load_dotenv


# 간단하게 마크다운으로 출력해주는 구문
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def gemini_generate(model, config, gen_count = 10000):
    labels = ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']

    generate_list = []

    for idx, label in enumerate(labels):
        goal = f"""input: 위를 참고하여 {label} 관련 한국어 기사제목을 10개 이상 생성해줘, 반드시 문장과 문장 사이에 '/'을 넣어 구분하여 출력해줘 \noutput:"""
        count = 0
        while count < gen_count:
            print('------', label, '-', count, '------')
            try:
                response = model.generate_content(few_shot_text+goal, generation_config=config)
                # print(response.text)
                for sentence in response.text.split('/'):
                    generate_list.append({'ID': str(idx)+'_'+str(count), 'text': sentence, 'target': idx, 'url': '', 'date': ''})
                    count += 1
                    if count >= gen_count:
                        break
                response = ''
            except:
                print('------error occured------')
                continue
        
    df = pd.DataFrame(generate_list, columns=['ID', 'text', 'target', 'url', 'date'])
    df = cleaning(df)
    return df

def cleaning(df):
    print('before cleaning:', len(df))
    sentences = df['text'].tolist()
    drop_list = []
    for idx, sentence in enumerate(sentences):
        if type(sentence) != str:
            drop_list.append(idx)
        elif len(sentence) < 10 or len(sentence) > 256:
            drop_list.append(idx)
        elif 'input' in sentence or 'output' in sentence:
            drop_list.append(idx)
    new_df = df.drop(drop_list)
    new_df = new_df.drop_duplicates(['text'])
    new_df['text'] = [sentence.strip().replace('\n', ' ').replace('"', ' ') for sentence in new_df['text'].tolist()]
    print('after cleaning:', len(new_df))
    return new_df

def parse_arguments() :
        
    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--gen_count', type=str, default=1000)
    parser.add_argument('--output_dir', type=str, default='../data/gemini_generate.csv')
    
    args = parser.parse_args()

    return args

##############################################################################################################
args = parse_arguments()

load_dotenv(verbose=True)
    
api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

config = {
    "max_output_tokens": 2048,
    "temperature": 0.9,# 0~1 높을수록 다양한 생성 결과
    # "top_p" : 0.5 
}

few_shot_text = """
input: 네이버 뉴스기사들을 참고하여 '정치'와 관련된 한국어 기사제목을 생성해줘
output: 황총리 안보위기 평화통일 전기되도록 할 것/한국 계파갈등 재연 조짐…연이틀 비대위·친박 신경전/불법·허위사실 vs 적법…곳곳서 여론조사 논란
input: 네이버 뉴스기사들을 참고하여 'IT과학'와 관련된 한국어 기사제목을 생성해줘
output: 갤럭시S8 중동 상륙…28일 아랍에미리트 출시/게시판 SK텔링크 어르신 특화폰 효도의 신 2탄 출시/네이버 제2 IDC 클라우드에 특화…이해진 전세계 데이터 거점종합
input: 네이버 뉴스기사들을 참고하여 '스포츠'와 관련된 한국어 기사제목을 생성해줘
output: '불펜 데이' 다저스, 홈런 4방으로 샌디에이고 제압/현대건설 정규리그 기준 팀 최다 14연패…이번 시즌은 8.../전북 단장 김민재 이적 베이징·톈진 중 신중하게 선택
input: 네이버 뉴스기사들을 참고하여 '경제'와 관련된 한국어 기사제목을 생성해줘
output: 국내 주식형 펀드 일주일 만에 자금 순유입/코스메카코리아 200억원 전환사채 발행 결정/강남 아파트 고분양가 경쟁 후끈…다음 주자는
input: 네이버 뉴스기사들을 참고하여 '생활문화'와 관련된 한국어 기사제목을 생성해줘
output: 추석 보름달 수도권·전북에서만 빼꼼…5∼6일 전국 비/동해 탄생의 비밀 간직한 비경 강릉 '바다부채길' 6월 정식 개방/광양 37도 폭염 올들어 최고…강·바다·산 인산인해종합
input: 네이버 뉴스기사들을 참고하여 '세계'와 관련된 한국어 기사제목을 생성해줘
output: 폼페이오 "이란, 협상 테이블로 나와야"…무인기 격추 재확인/알자지라 이란 폭발물수거 美영상 사건 10시간 뒤 촬영/중국 비난 트럼프 트위터 中네티즌 맹공에 혼쭐
input: 네이버 뉴스기사들을 참고하여 '사회'와 관련된 한국어 기사제목을 생성해줘
output: 정부 편법증여·대출의심거래 엄정대응…이달말 조사결과 발표종합/제2의 블랙 스완을 피하는 법…행동과 책임의 균형/경찰 손석희 출석 일정 조율 중…피혐의자 신분
"""

df = gemini_generate(model, config, args.gen_count)

df.to_csv(args.output_dir, index=False)
      
