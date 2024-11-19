#!/bin/bash

csv_path=$1 # wgere to save scv_path
save_dir=$2 # where to save result
task=${3:-regression}
adjacency_path=$4
distance_path=$5
coulomb_path=$6
features_path=$7
target_columns=$8
smiles_columns=$9
id=${10:-sample}

log_dir="./log/$id"
mkdir -p $log_dir


# Print the values
echo "CSV Path: $csv_path"
echo "Save Directory: $save_dir"
echo "Task: $task"
echo "Adjacency Path: $adjacency_path"
echo "Distance Path: $distance_path"
echo "coulomb_path: $coulomb_path"
echo "Features Path: $features_path"
echo "target column: " $target_columns
echo "smiles columns: $smiles_columns"


log_path="$log_dir/train.log"
echo "log file will export to $log_path"

python train.py --save_dir $save_dir \
    --data_path $csv_path \
    --target_columns $target_columns \
    --smiles_columns $smiles_columns \
    --dataset_type $task \
    --bond_fast_attention \
    --atom_attention \
    --adjacency \
    --adjacency_path $adjacency_path \
    --distance \
    --distance_path $distance_path \
    --coulomb \
    --coulomb_path $coulomb_path \
    --normalize_matrices \
    --features_path $features_path \
    --split_type random \
    --no_features_scaling \
    --num_folds 5 \
    --save_preds \
    --gpu 0 > $log_path
