#!/bin/bash

# data_path="data/mydata/oxime_ester.smi"
data_path="data/mydata/chromo_smiles_new.smi"
# tokennizer_path="data/tokenizers/zincBPETokenizer_new.json"
tokennizer_path="data/tokenizers/oximeBPETokenizer.json"
save_path="data/results/oxime_ester_symm_oximenum_aromaticnum_hetero_maxringsize_OximePropReward_4"
tokenizer="BPE"
# reward_fns_name="Symmetry NumOximeActivationReward FusedRing NumAromaticHeterocycles MaxRingSize OximePropReward"
reward_fns_name="Symmetry NumOximeActivationReward FusedRing MaxRingSize OximePropReward"
# reward_fns_name="NumOximeActivationReward FusedRing NumAromaticHeterocycles MaxRingSize OximePropReward"
model_path="data/results/oxime_fine_tune_random/NL_model/epoch_6.pt"
# multipliers=("lambda x: x" "lambda x: x" "lambda x: x" "lambda x: x" "lambda x: x" "lambda x: x")
multipliers=("lambda x: x" "lambda x: x" "lambda x: x" "lambda x: x" "lambda x: x")
predictor_paths="None None None None None"
batch_size=256
# eval_max_len=200
eval_max_len=300
# rl_max_len=200
rl_max_len=300
eval_size=25000
rl_epochs=1000
eval_steps=10
python MolGen/main.py --do_eval --load_pretrained --pretrained_path $model_path --save_path $save_path --dataset_path $data_path --tokenizer $tokenizer --tokenizer_path $tokennizer_path --reward_fns $reward_fns_name --multipliers "${multipliers[@]}" --predictor_paths $predictor_paths --batch_size $batch_size --eval_max_len $eval_max_len --eval_size $eval_size --eval_steps $eval_steps --rl_epochs $rl_epochs --rl_max_len $rl_max_len