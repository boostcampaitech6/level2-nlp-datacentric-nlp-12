#!/bin/bash

# command : ./data_pipeline [input_dir] [output_dir] [G2P ratio] [seed]
# example : ./data_pipeline ../../data/train.csv ../../data/final.csv 0.1 486

# 허깅페이스 필터링
python3 hug_filter.py --target_dir $1

# 밸런싱
python3 balancing.py --seed $4

# G2P 적용한 후 원 데이터와 concat
python3 G2P.py --output_dir $2 --g2p_ratio $3

# 중간 부산물들 삭제 (선택)
# rm ../data/relabel.csv
# rm ../data/balanced.csv
# rm ../data/g2ped.csv