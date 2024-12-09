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
bash preprocess_train.sh "data/oxime_and_other/oxime_and_other.csv" "./features/oxim" "smiles" "oximeester"
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

We Trained T1 enery with following code.

```
bash train.sh data/oxime_and_other/oxime_and_other.csv feature/oxim regression feature/oxim/adj.npz feature/oxim/clb.npz feature/oxim/dist.npz feature/oxim/rdkit.npz T1 smiles oxim
```

### 3. Visualize Results

```
bash see_attention.sh {csv_path} {ckpt_dir} {pred_path} {viz_save_dir} {adjacency_path} {distance_path} {coulomb_path} {features_path}
```

# Reference

Referenced from [ABT-MPNN: An atom-bond transformer-based message passing neural network for molecular property prediction](https://doi.org/10.1186/s13321-023-00698-9).
