# ABT-MPNN: An atom-bond transformer-based message passing neural network for molecular property prediction

### Requirements

- jq
  A package for handling JSON
  Installation methods:

  ```
  # Ubuntu
  sudo apt-get jq
  # Mac (Homebrew)
  brew install jq
  ```

## Usage

### 0. Python Environment Setup

Set up using either pip or conda environment

```
pip install -r requirements.txt
conda --env create -f environment.yml
```

After setup, specify the path to your Python environment in setting.json

`setting.json`

```
{
    "python_path": "/path/to/your/python"
}
```

### 1. Preprocess Data

Performs data preprocessing, including feature extraction

**About Arguments**

```
csv_path=$1 <- path to CSV file
save_dir=$2 <- directory to save output
smiles_column=$3 <- name of SMILES column in CSV file
task_name=$4 <- task name
```

`task_name` specifies the log output location. If `task_name="sample"`, logs will be output to `log/sample`

**Execution Command**

```
bash preprocess_train.sh {csv_path} {save_dir} {smiles_column} {task_name}
```

**Example**

```
bash preprocess_train.sh "./data/freesolv.csv" "./features/freesolv" "smiles" "freesolv"
```

### 2. Training

Training command that supports both classification and regression

**About Arguments**

```
{csv_path} {save_dir} {task} {adjacency_path} {distance_path} {coulomb_path} {features_path} {target_columns} {smiles_columns} {id}
```

**Execution Command**

```
bash train.sh {csv_path} {save_dir} {task} {adjacency_path} {distance_path} {coulomb_path} {features_path} {target_columns} {smiles_columns} {id}
```

**Example**

```
bash train.sh data/qm9.csv feature/qm9 regression feature/qm9/adj.npz feature/qm9/clb.npz feature/qm9/dist.npz feature/qm9/rdkit.npz homo smiles qm6
```

Sample data download: https://drive.google.com/file/d/1reNIj65nKHPblIM00_gpbQqmWrmNZc1k/view?usp=drive_link

### 3. Visualize Results

```
bash see_attention.sh {csv_path} {ckpt_dir} {pred_path} {viz_save_dir} {adjacency_path} {distance_path} {coulomb_path} {features_path}
```

# Reference

Referenced from [ABT-MPNN: An atom-bond transformer-based message passing neural network for molecular property prediction](https://doi.org/10.1186/s13321-023-00698-9).
![](https://github.com/LCY02/ABT-MPNN/blob/master/docs/architecture.png?raw=true)
