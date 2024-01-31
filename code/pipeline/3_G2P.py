
import argparse
import pandas as pd
from tqdm import tqdm
import random
from g2pk import G2p

def apply_g2p(df, frac) :
    
    sample_size = int(frac * len(df))
    sample_indices = sorted(random.sample(range(len(df)), sample_size))
    
    g2p = G2p()

    for idx in tqdm(sample_indices) :
        df.loc[idx, 'text'] = g2p(df.loc[idx, 'text'])

def parse_arguments() :
    
    parser = argparse.ArgumentParser(description='Argparse')

    parser.add_argument('--target_dir', type=str, default='../data/balanced.csv')
    parser.add_argument('--output_dir', type=str, default='../data/g2ped.csv')
    parser.add_argument('--g2p_ratio', type=float, default=0.1)
    parser.add_argument('--train_concat', type=str, default='../data/train_relabel.csv')
    
    args = parser.parse_args()

    return args


# G2P 적용
args = parse_arguments()
df_org = pd.read_csv(args.target_dir)

df_g2p = df_org.copy()
apply_g2p(df_g2p, args.g2p_ratio)

# log 출력
joined_df = pd.merge(df_org, df_g2p, on='text', how='inner')
print("Total {} data was G2Ped out of {} data.".format( len(df_org) - len(joined_df), len(df_org) ) )

df_g2p = df_g2p[["text", "target"]]

if args.train_concat :
    train_relabel = pd.read_csv(args.train_concat)
    train_relabel = train_relabel[["text", "target"]]
    df_g2p = pd.concat([df_g2p, train_relabel], axis=0)

df_g2p.to_csv(args.output_dir, index=False)
