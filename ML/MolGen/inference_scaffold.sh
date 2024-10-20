#!/bin/bash

datime=$(date "+%Y%m%d%H%M%S")
dataset_path="data/mydata/oxime_ester.smi"
model_path="/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/data/results/oxime_ester_symm_oximenum_aromaticnum_hetero_maxringsize_OximePropReward_3_GPT_Layers_4_Heads_8_Emb_512_Dmodel_1024_chromo_smiles_new_RlBatch_500_RlEpochs_1000_Reward_MultiReward_Scaffold_False_discount_0.99/mid_RL/step_50/model.pt"
tokenizer_path="data/tokenizers/zincBPETokenizer_new.json"
tokenizer="BPE"
batch_size=256
save_path="data/results/generated_smiles/$datime.csv"
# eval_size=1000000
eval_size=10000000
echo "save_path: $save_path"
python MolGen/inference.py --use_scaffold --dataset_path $dataset_path --pretrained_path $model_path --tokenizer $tokenizer --tokenizer_path $tokenizer_path --batch_size $batch_size --save_path $save_path --eval_size $eval_size
