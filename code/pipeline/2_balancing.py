
import argparse
import pandas as pd

def parse_arguments() :
    
    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--target_dir', type=str, default='../../data/relabel.csv')
    parser.add_argument('--output_dir', type=str, default='../../data/balanced.csv')
    parser.add_argument('--data_per_label', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=486)
    
    args = parser.parse_args()

    return args


args = parse_arguments()


### 파일을 받아서
df_sure = pd.read_csv(args.target_dir)

### count_value로 최소 레이블을 찾아낸 후
minsample = df_sure["target"].value_counts().iloc[-1]

# 지정해 준 경우
if args.data_per_label > 0 :

    # 뽑을 수 있는 최대를 넘긴 경우 조정해줌    
    if args.data_per_label > minsample :
        print("data per label exceeds max number. Adjusting to available max {}...".format(minsample))
        data_per_label = minsample
    
    # 아닌 경우 그대로 지정
    else :
        data_per_label = args.data_per_label

# 지정해주지 않으면 그냥 할 수 있는 최대로 뽑음
else :
    data_per_label = minsample


### 그거에 맞춰서 샘플링.
df_list = []
for target in range(7) :
    df_list.append(df_sure[df_sure['target'] == target].sample(data_per_label, random_state=args.seed))

# 합쳐서 저장
df_balance = pd.concat(df_list)
df_balance.to_csv(args.output_dir, index=False)

