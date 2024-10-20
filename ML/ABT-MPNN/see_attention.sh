#!/bin/bash

csv_path=$1 # wgere to save scv_path
ckpt_path=$2 # checpint directory
pred_path=$3 # predict result path
viz_save_dir=$4 # where to save vizualized result
adjacency_path=$5
distance_path=$6
coulomb_path=$7
features_path=$8
smiles_column=$9

# Print the values
echo "CSV Path: $csv_path"
echo "ckpt dirrecotry: $ckpt_path"
echo "pred path: $pred_path"
echo "viz_save_dir $viz_save_dir"
echo "Adjacency Path: $adjacency_path"
echo "Distance Path: $distance_path"
echo "coulomb path: $coulomb_path"
echo "Features Path: $features_path"
echo "Smiles Columnb: $smiles_column"

# load python path
PYTHON_PATH=$(jq -r '.python_path' setting.json)
echo "Your python path: $PYTHON_PATH"

$PYTHON_PATH see_attention.py --test_path $csv_path \
    --checkpoint_path $ckpt_path \
    --preds_path $pred_path \
    --viz_dir $viz_save_dir \
    --adjacency_path $adjacency_path \
    --distance_path $distance_path \
    --coulomb_path $coulomb_path \
    --features_path $features_path \
    --normalize_matrices --no_features_scaling \
    --smiles_column $smiles_column