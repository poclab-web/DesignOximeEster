from mychemprop.utils import load_checkpoint
from mychemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
# from mychemprop.args import PredictArgs, TrainArgs
from mychemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from mychemprop.data.utils import get_data_from_smiles_and_graph
from mychemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from mychemprop.features import set_extra_atom_fdim, set_extra_bond_fdim
from mychemprop.train import predict
from mychemprop.atomic_matrices import mol2matrix
from mychemprop.features import get_available_features_generators, get_features_generator, load_features, save_features

from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


import torch
import numpy as np
from typing import List, Optional, Union
import json
from rdkit import Chem
from tqdm import tqdm


class Args:
    def __init__(self, json_path):
        self.json_path = json_path
        self.args = self.load_json(json_path)
        # set as fields
        for key, value in self.args.items():
            setattr(self, key, value)
    
    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        return json_data


class ABTMPNN:
    def __init__(self, checkpoint_path, args_path, device=None):
        if device is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = device
        self.device = torch.device(device_name)
        self.checkpoint_path = checkpoint_path

        self.args = Args(args_path)

        update_prediction_args(predict_args=self.args, train_args=self.args)

        print('Loading model from {}'.format(checkpoint_path))
        self.model = load_checkpoint(checkpoint_path, self.device, None)
        self.model.eval()
        print('mode loaded')
    

    def get_smiles(self, smiles:list):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        return [smiles, mols]
    

    def make_graph(self, smiles:list):
        # exit()
        graph = mol2matrix(smiles, self.args)
        if self.args.coulomb:
            clb = graph.get_coulomb()[1:]
            clb = np.array(clb, dtype=object)
        else:
            clb = None
        if self.args.distance:
            dist = graph.get_distance()[1:]
            dist = np.array(dist, dtype=object)
            # print('tmp_dist.shape', tmp_dist.shape)
        else:
            dist = None
        if self.args.adjacency:
            adj = graph.get_adjacency()[1:]
            adj = np.array(adj, dtype=object)
        else:
            adj = None
        return clb, dist, adj
    
    def make_features(self, feature_generator:str):
        if feature_generator is not None:
            if type(self.args.features_generator) is not List:
                feature_generator = [self.args.features_generator]
        else:
            feature_generator = None
        return feature_generator

    def make_need_data(self, smiles:list):
        # smiles_convert_with_mol = []
        # for s in smiles:
        #     smiles_convert_with_mol.append(self.get_smiles(s))
        clb, dist, adj = self.make_graph(smiles)
        feature_generator = self.make_features(self.args.features_generator)
        return clb, dist, adj, feature_generator
    

    def is_able_to_predict(self, smiles:str):
        # 入力したSmilesが予測可能か判定する
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        
        if smiles is not None:
            try:
                generator = rdNormalizedDescriptors.RDKit2DNormalized()
                features = generator.process(smiles)[1:]
                if features is not None:
                    return True
                else:
                    return False
            except:
                import traceback
                print(traceback.format_exc())
                return False
        else:
            None
            



    def preprocess_predict(self, smiles: List[List[str]], num_tasks:int):
        # update_prediction_args(predict_args=args, train_args=self.args)
        # args: Union[PredictArgs, TrainArgs]
        clb, dist, adj, feature_generator = self.make_need_data(smiles)

        if self.args.atom_descriptors == 'feature':
            set_extra_atom_fdim(self.args.atom_features_size)

        if self.args.bond_features_path is not None:
            set_extra_bond_fdim(self.args.bond_features_size)

        full_data = get_data_from_smiles_and_graph(
            smiles=smiles,
            coulomb_matrix=clb,
            distance_matrix=dist,
            adjacency_matrix=adj,
            skip_invalid_smiles=False,
            features_generator=feature_generator
        )

        print('Validating SMILES')
        full_to_valid_indices = {}
        valid_index = 0
        for full_index in range(len(full_data)):
            if all(mol is not None for mol in full_data[full_index].mol):
                full_to_valid_indices[full_index] = valid_index
                valid_index += 1
        
        # make dataset
        print('Creating test data', full_data)
        test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

        # Edge case if empty list of smiles is provided
        if len(test_data) == 0:
            return [None] * len(full_data)

        print(f'Test size = {len(test_data):,}')

        # Predict with each model individually and sum predictions
        if self.args.dataset_type == 'multiclass':
            sum_preds = np.zeros((len(test_data), num_tasks, self.args.multiclass_num_classes))
        else:
            sum_preds = np.zeros((len(test_data), num_tasks))

        # Create data loader
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers
        )

        return test_data_loader, test_data


    def predict(self, smiles:list):
        num_tasks = self.args.num_tasks
        
        pred_data_loader, pred_data = self.preprocess_predict(smiles, num_tasks)

        scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = load_scalers(self.checkpoint_path)

        # Normalize features
        if self.args.features_scaling or self.args.atom_descriptor_scaling or self.args.bond_feature_scaling:
            pred_data.reset_features_and_targets()
            if self.args.features_scaling:
                pred_data.normalize_features(features_scaler)
            if self.args.atom_descriptor_scaling and self.args.atom_descriptors is not None:
                pred_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if self.args.bond_feature_scaling and self.args.bond_features_size > 0:
                pred_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

        preds = predict(
            model=self.model,
            data_loader=pred_data_loader,
            disable_progress_bar=False,
            scaler=scaler
        )

        return preds

import math
def inverse_log_transform(value: float) -> float:
    return math.exp(value) - 1e-10

if __name__ == "__main__":
    import time
    json_path = "./log/oxime_and_other_transed/args.json"
    # args = Args(json_path)

    import pandas as pd
    import sys
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    all_smiles = df['SMILES'].values.tolist()
    # print('smiles_list', smiles_list)
    # smiles_list = [[s] for s in all_smiles]
    
    def split_smiles_to_batches(smiles, batch_size):
        return [smiles[i:i + batch_size] for i in range(0, len(smiles), batch_size)]


    checkpoint_path = "./log/oxime_and_other_transed/fold_1/model_0/model.pt"

    model = ABTMPNN(checkpoint_path, json_path)

    smiles_batches = split_smiles_to_batches(all_smiles, 256)
    
    T1, S1, S1_osc = [], [], []
    for smiles_batch in tqdm(smiles_batches, total=len(smiles_batches)):
        pred_batch = model.predict(smiles_batch)
        for pred in pred_batch:
            T1.append(pred[0])
            S1.append(pred[1])
            S1_osc.append(inverse_log_transform(pred[2]))
        
    df["T1"] = T1
    df["S1"] = S1
    df["S1_osc"] = S1_osc
    
    df.to_csv("pred.csv", index=False)
