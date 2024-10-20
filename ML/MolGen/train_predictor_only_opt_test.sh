#!/bin/bash

data_path="data/mydata/pubchem_aldrich_filter.smi"
tokennizer_path="data/tokenizers/zincBPETokenizer.json"
save_path="data/results/test"
tokenizer="BPE"
reward_fns_name="Symmetry FusedRing MaxRingSize NumOximeActivationReward"
model_path="data/results/oxime_fine_tune/NL_model/epoch_2.pt"
multipliers=("lambda x: x"  "lambda x: x" "lambda x: x" "lambda x: x")
predictor_paths="None None None None"
batch_size=192
eval_max_len=10
eval_size=25
eval_steps=10
rl_epochs=2
python MolGen/main.py --do_eval --load_pretrained --pretrained_path $model_path --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "${multipliers[@]}"  --predictor_paths $predictor_paths --batch_size $batch_size --eval_max_len $eval_max_len --rl_epochs $rl_epochs
