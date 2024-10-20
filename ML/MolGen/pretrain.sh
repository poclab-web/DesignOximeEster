#!/bin/bash

data_path="data/mydata/pretrain_data.smi"
tokennizer_path="data/tokenizers/oximeBPETokenizer.json"
save_path="data/results/pretrain_aldrich_randomize"
tokenizer="BPE"
reward_fns_name="Symmetry"
multipliers=("lambda x: 2*x")
predictor_paths="None"
batch_size=256
eval_max_len=200
eval_size=100000
eval_steps=100
epochs=10
python MolGen/main.py --do_train --epochs $epochs --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "${multipliers[@]}" --predictor_paths $predictor_paths --batch_size $batch_size --eval_max_len $eval_max_len