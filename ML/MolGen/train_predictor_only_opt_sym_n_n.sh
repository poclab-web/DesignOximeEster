#!/bin/bash

data_path="data/mydata/pubchem_aldrich_filter.smi"
tokennizer_path="data/tokenizers/zincBPETokenizer.json"
save_path="data/results/aldrich_pretrain"
tokenizer="BPE"
model_path="data/results/chromo_fine_tune/NL_model/epoch_2.pt"
reward_fns_name="Symmetry FusedRing MaxRingSize"
multipliers=("lambda x: x" "lambda x: x" "lambda x: x")
predictor_paths="None None None"
batch_size=256
eval_max_len=200
eval_size=100000
eval_steps=100
python MolGen/main.py --do_eval --load_pretrained --pretrained_path $model_path --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "${multipliers[@]}" --predictor_paths $predictor_paths --batch_size $batch_size --eval_max_len $eval_max_len 