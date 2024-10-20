#!/bin/bash

data_path="data/mydata/chromo_smiles_new.smi"
tokennizer_path="data/tokenizers/oximeBPETokenizer.json"
save_path="data/results/OximeEster8"
tokenizer="BPE"
reward_fns_name="Symmetry NumOximeActivationReward FusedRing MaxRingSize OximePropReward SmilesLength NumAromaticHeterocycles"
model_path="data/results/oxime_fine_tune_random/NL_model/epoch_2.pt"
multipliers=("lambda x: x" "lambda x: x" "lambda x: x" "lambda x: x" "lambda x: x" "lambda x: 0.01*x" "lambda x: x")
predictor_paths="None None None None None None None"
batch_size=256
eval_max_len=300
rl_max_len=300
eval_size=10000
rl_epochs=100000
eval_steps=100
python MolGen/main.py --do_eval --load_pretrained --pretrained_path $model_path --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "${multipliers[@]}" --predictor_paths $predictor_paths --batch_size $batch_size --eval_max_len $eval_max_len --eval_size $eval_size --eval_steps $eval_steps --rl_epochs $rl_epochs --rl_max_len $rl_max_len
