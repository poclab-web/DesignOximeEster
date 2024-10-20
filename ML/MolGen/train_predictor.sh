#!/bin/bash

data_path="data/mydata/chromo_smiles.smi"
tokennizer_path="data/tokenizers/zincCharTokenizer.json"
save_path="data/results/chromo_pretrain"
tokenizer="BPE"
reward_fns_name="Symmmetry"
batch_size=128
python MolGen/main.py --do_train --do_eval --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "lambda x: x" --batch_size $batch_size