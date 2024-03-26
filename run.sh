#!/bin/bash

# * laptop

# * DAGCN
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name 'dagcn' --dataset 'laptop' --seed 1000 --num_epoch 50 --vocab_dir './dataset/Laptops_corenlp' --alpha 0.7 --beta 0.9 --gama 0.3
# * DAGCN with Bert
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name 'dagcnbert' --dataset 'laptop' --seed 1000  --bert_lr 2e-5 --num_epoch 15 --hidden_dim 768 --max_length 100 --alpha 1.0 --beta 0.2 --gama 1.6


# * restaurant

# * DAGCN
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name 'dagcn' --dataset restaurant --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Restaurants_corenlp --alpha 0.7 --beta 0.3 --gama 0.8
# * DAGCN with Bert
CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name 'dagcnbert' --dataset restaurant --seed 1000 --bert_lr 2e-5 --num_epoch 1 --hidden_dim 768 --max_length 100 --alpha 0.8 --beta 0.6 --gama 1.5


# * twitter

# * DAGCN
# CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name 'dagcn' --dataset twitter --seed 1000 --num_epoch 50 --vocab_dir ./dataset/Tweets_corenlp --alpha 0.2 --beta 0.6 --gama 0.2
# * DAGCN with Bert
CUDA_VISIBLE_DEVICES=0 python ./train.py --model_name 'dagcnbert' --dataset twitter --seed 1000 --bert_lr 2e-5 --num_epoch 1 --hidden_dim 768 --max_length 100 --alpha 0.4 --beta 0.3 --gama 1.3
