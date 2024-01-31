#!/bin/bash

# 허깅페이스 필터링
python3 hug_filter.py --target_dir ../data/aihub_train.csv

# 밸런싱
python3 balancing.py --seed 486

# G2P 적용한 후 원 데이터와 concat
python3 G2P.py --output_dir ../data/final.csv --g2p_ratio 0.1

# 중간 부산물들 삭제
# rm ../data/relabel.csv
# rm ../data/balanced.csv
# rm ../data/g2ped.csv