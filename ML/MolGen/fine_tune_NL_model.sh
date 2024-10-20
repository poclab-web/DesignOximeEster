#!/bin/bash

data_path="data/mydata/oxime_ester.smi"
pretrained_path="data/results/pretrain_aldrich_randomize/NL_model/epoch_9.pt"
tokennizer_path="data/tokenizers/oximeBPETokenizer.json"
save_path="data/results/oxime_fine_tune_random"
tokenizer="BPE"
reward_fns_name="Symmetry FusedRing MaxRingSize"
model_path="data/results/oxime_fine_tune/NL_model/epoch_2.pt"
multipliers=("lambda x: x" "lambda x: x" "lambda x: x")
predictor_paths="None None None"
batch_size=256
eval_max_len=200
eval_size=100000
eval_steps=100
epochs=100
python MolGen/main.py --do_train --epochs $epochs --load_pretrained --pretrained_path $pretrained_path --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "${multipliers[@]}" --predictor_paths $predictor_paths --batch_size $batch_size --eval_max_len $eval_max_len 