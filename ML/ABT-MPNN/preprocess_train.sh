#!/bin/bash

csv_path=$1 # wgere to save scv_path
save_dir=$2 # where to save result
smiles_column=$3 # smiles column
task_name=$4

log_dir="./log/$task_name"
mkdir -p $log_dir


# load python path
PYTHON_PATH=$(jq -r '.python_path' setting.json)
echo "Your python path: $PYTHON_PATH"

# atom featurization
log_path="$log_dir/save_atom_features.log"
$PYTHON_PATH save_atom_features.py --data_path $csv_path --smiles_column $smiles_column --save_dir $save_dir --adjacency --coulomb --distance > $log_path

# save rdkit feature
log_path="$log_dir/save_features.log"
save_path="$save_dir/rdkit"
$PYTHON_PATH save_features.py --data_path $csv_path --smiles_column $smiles_column --save_path $save_path --features_generator rdkit_2d_normalized > $log_path