#!/bin/bash

data_path="data/mydata/chromo_smiles.smi"
tokennizer_path="data/tokenizers/zincBPETokenizer.json"
save_path="data/results/chromo_pretrain"
tokenizer="BPE"
reward_fns_name="Symmetry FusedRing NumAromaticHeterocycles"
model_path="data/models/gpt_pre_rl_zinc_bpe.pt"
multipliers="lambda x: x lambda x: x** lambda x: x**"
predictor_paths="None None None"
batch_size=192
eval_max_len=100
eval_size=1000
eval_steps=10
python MolGen/main.py --do_eval --load_pretrained --pretrained_path $model_path --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "lambda x: x" --predictor_paths $predictor_paths --batch_size $batch_size --eval_max_len $eval_max_len 