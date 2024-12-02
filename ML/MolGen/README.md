# Installtion Guide

Install the conda enviroment using the following command:
`conda env create -f environment.yml`

# pretrain dataset

We use Aldrich dataset for pretrain

```
data/mydata/pretrain_data.smi
```

# oxime ester dataset

We use fine oxime ester dataset for fine tuning

```
data/mydata/oxime_ester.smi
```

# Running

To train a language model and then perform reinforcement learning optimization run:
`python3 MolGen/main.py --do_train --do_eval --dataset_path ./data/gdb/gdb13/gdb13.smi --tokenizer Char --tokenizer_path ./data/tokenizers/gdb13CharTokenizer.json --reward_fns QED --multipliers "lambda x: x" --batch_size 256`

To only perform reinfocement learning optimization with a pretrained language model run:
`python3 MolGen/main.py --load_pretrained --pretrained_path ./data/models/gpt_pre_rl_gdb13.pt --do_eval --dataset_path ./data/gdb/gdb13/gdb13.smi --tokenizer Char --tokenizer_path ./data/tokenizers/gdb13CharTokenizer.json --reward_fns QED --multipliers "lambda x: x" --batch_size 256`

# Reference

https://github.com/eyalmazuz/MolGen
