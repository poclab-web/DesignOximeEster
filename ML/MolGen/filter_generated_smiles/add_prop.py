import pandas as pd
from rdkit import Chem
import metrics as filter_funcs
import chem_utils.mol as mol_utils

# remove rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import sys, os
abtmpnn_dir = os.path.join(os.path.dirname(__file__), '../../ABT-MPNN')
sys.path.append(abtmpnn_dir)
from ABTMPNN import ABTMPNN


class AddPropToDF:
    def __init__(self, **kwargs):
        self.abtmpnn = None
        self.init_abtmpnn(kwargs['abtmpnn_checkpoint_path'], kwargs['abtmpnn_args_path'])
    
    def init_abtmpnn(self, checkpoint_path, args_path):
        self.abtmpnn = ABTMPNN(checkpoint_path, args_path, 'cpu')

    def read_csv(self, csv_path: str):
        if not csv_path.endswith('.csv'):
            raise ValueError('csv_path must be csv file path.')
        df = pd.read_csv(csv_path)
        return df
    
    def remove_invalid_smiles(self, df, smiles_col):
        df = df[df[smiles_col].apply(mol_utils.smiles2mol2smiles) != None]
        df = df.dropna()
        return df
    

    def add_prop2df_with_multiprop(self, df, smiles_col, func):
        all_smiles = df[smiles_col].values.tolist()
        all_props = func(self.abtmpnn, all_smiles)
        for prop in all_props:
            df[prop] = all_props[prop]
        return df
    
    def add_prop2df_with_single_prop(self, df, smiles_col, prop_col, func):
        # apply func to each smiles
        df[prop_col] = df[smiles_col].apply(func)
        return df

    def __call__(self, df:pd.DataFrame, smiles_col='Smiles', save_path=None):

        df = self.remove_invalid_smiles(df, smiles_col)
        
        # oxime num
        df = self.add_prop2df_with_single_prop(df, smiles_col, 'oxime_num', filter_funcs.num_oxime_activation)

        # ring num
        df = self.add_prop2df_with_single_prop(df, smiles_col, 'max_fused_ring_size', filter_funcs.get_max_fused_ring_size)

        # symmetry
        df = self.add_prop2df_with_single_prop(df, smiles_col, 'symmetry', filter_funcs.symmetry)

        # abtmpnn
        # df = self.add_prop2df_with_multiprop(df, smiles_col, filter_funcs.abtmpnn_predict)

        if save_path is not None:
            df.to_csv(save_path, index=False, header=False)
        return df
        


if __name__ == "__main__":
    abtmpnn_checkpoint_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/fold_1/model_0/model.pt"
    abtmpnn_args_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/args.json"
    addprop = AddPropToDF(abtmpnn_checkpoint_path=abtmpnn_checkpoint_path, abtmpnn_args_path=abtmpnn_args_path)
    csv_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/data/results/generated_smiles/20240112011535.csv"
    save_scv_path = "./add_prop_20240112011535.csv"

    df = pd.read_csv(csv_path)
    addprop(df, save_path=save_scv_path)

