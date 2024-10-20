import copy
from datetime import datetime
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
# from parso import parse
import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
# lg = RDLogger.logger()
# lg.setLevel(RDLogger.CRITICAL)

from sklearn.model_selection import train_test_split
import torch

from src.datasets.get_dataset import get_dataset
from src.datasets.bs1_dataset import BS1Dataset
from src.models.model_builder import get_model 
from src.models.gpt import GPTValue
from src.models.bert import Bert, BertConfig
from src.tokenizers.CharTokenizer import CharTokenizer
from src.tokenizers.BPETokenizer import BPETokenizer
from src.train.train import Trainer, PredictorTrainer
from src.train.evaluate import generate_smiles, generate_smiles_scaffolds, get_stats, gen_till_train
from src.train.reinforcement import policy_gradients
from src.utils.reward_fn import QEDReward, get_reward_fn
from src.utils.utils import TaskOpt, get_max_smiles_len
from src.utils.utils import parse_arguments
from src.utils.mol_utils import filter_invalid_molecules

torch.autograd.set_detect_anomaly(True)

def main():

    # start = time.time()

    args = parse_arguments()

    device = torch.device(args.device)

    print(args.device)
    print(args.reward_fns)
    print('multiplers', args.multipliers)
    
    max_smiles_len = get_max_smiles_len(args.dataset_path) + 50
    #max_smiles_len = 256
    print(f'max smiles len: {max_smiles_len}')
    # print(f'{max_smiles_len=}')

    if args.tokenizer == "Char":
        tokenizer = CharTokenizer(args.tokenizer_path, args.dataset_path)

    elif args.tokenizer == "BPE":
        tokenizer = BPETokenizer(args.tokenizer_path, args.dataset_path, vocab_size=500)

    else:
        raise ValueError("Tokenizer type not supported")

    dataset = get_dataset(data_path=args.dataset_path,
                          tokenizer=tokenizer,
                          use_scaffold=args.use_scaffold,
                          max_len=max_smiles_len)

    model = get_model(args.model,
                      n_embd=args.n_embd,
                      d_model=args.d_model,
                      n_layers=args.n_layers,
                      num_heads=args.num_heads,
                      vocab_size=tokenizer.vocab_size,
                      block_size=args.block_size,
                      proj_size=args.proj_size,
                      attn_dropout_rate=args.attn_dropout_rate,
                      proj_dropout_rate=args.proj_dropout_rate,
                      resid_dropout_rate=args.resid_dropout_rate,
                      padding_idx=tokenizer.pad_token_id).to(device)

    print(args.pretrained_path)
    model.load_state_dict(torch.load(args.pretrained_path))

    
    if args.use_scaffold:
        print("Using scaffolds")
        generated_smiles = generate_smiles_scaffolds(model=model,
                                                    tokenizer=tokenizer,
                                                    scaffolds=dataset.scaffolds,
                                                    temprature=args.temprature,
                                                    size=args.eval_size,
                                                    max_len=args.eval_max_len,
                                                    device=device)
    else:
        generated_smiles = generate_smiles(model=model,
                                           tokenizer=tokenizer,
                                           temprature=args.temprature,
                                           size=args.eval_size,
                                           max_len=args.eval_max_len,
                                           device=device)

    generated_smiles = filter_invalid_molecules(generated_smiles)

    valid_smiles = []
    for smiles in generated_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                valid_smiles.append(smiles)
        except:
            pass
    df = pd.DataFrame({'Smiles': valid_smiles})
    df.to_csv(args.save_path, index=False)

    print('normally finished inference')
    
    # end = time.time()

    # print('total time elapsed: ', end - start, ' sec')

if __name__ == "__main__":
    main()
