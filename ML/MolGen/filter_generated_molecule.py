import pandas as pd
import sys, os
abtmpnn_dir = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN"
sys.path.append(abtmpnn_dir)
from ABTMPNN import ABTMPNN
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

class Filter:
    def __init__(self, original_smiles_path):
        
        original_smiles = open(original_smiles_path, "r").readlines()
        mols = [Chem.MolFromSmiles(s) for s in original_smiles]
        self.original_fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
        
        
        CHECKPOINT_PATH = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/fold_1/model_0/model.pt"
        ARGS_PATH = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/args.json"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predic_model = ABTMPNN(CHECKPOINT_PATH, ARGS_PATH, DEVICE)
        self.target_t1, self.target_s1, self.target_s1_osc = 2.7, 2.5, 1.0
    
    def split_smiles_to_batches(self, smiles, batch_size):
        return [smiles[i:i + batch_size] for i in range(0, len(smiles), batch_size)]
    
    def predict(self, generated_smiles_df_path):
        df = pd.read_csv(generated_smiles_df_path)
        smiles = df["Smiles"].tolist()
        smiles_batches = self.split_smiles_to_batches(smiles, 256)
        
        all_preds = []
        T1, S1, S1_osc = [], [], []
        for smiles_batch in tqdm(smiles_batches, total=len(smiles_batches)):
            pred_batch = self.predic_model.predict(smiles_batch)
            for pred in pred_batch:
                T1.append(pred[0])
                S1.append(pred[1])
                S1_osc.append(pred[2])
        
        df["T1"] = T1
        df["S1"] = S1
        df["S1_osc"] = S1_osc
        
        return df
    
    
    def tanimoto_similarity(self, smiles1):
        mol1 = Chem.MolFromSmiles(smiles1)
        
        mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        
        tanimoto_similarity = DataStructs.BulkTanimotoSimilarity(mol1_fp, self.original_fp)
        
        max_tanimoto_similarity = max(tanimoto_similarity)
        
        return max_tanimoto_similarity
        
    
    
    def filtering(self, df):
        df = df[(df["T1"] > self.target_t1) & (df["S1"] > self.target_s1) & (df["S1_osc"] > self.target_s1_osc) & (df["Tanimoto_similarity"] < 0.5)]
        
        return df
    
    def __call__(self, generated_smiles_df_path):
        perdicted_df = self.predict(generated_smiles_df_path)
        perdicted_df.to_csv("pred_tmp.csv", index=False)
        
        tanimoto_similarity = perdicted_df["Smiles"].apply(self.tanimoto_similarity)
        perdicted_df["Tanimoto_similarity"] = tanimoto_similarity
        perdicted_df.to_csv("pred_tmp.csv", index=False)
        perdicted_df = self.filtering(perdicted_df)
        
        return perdicted_df
        
if __name__ == "__main__":
    original_df_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/data/mydata/oxime_ester.smi"
    filtering = Filter(original_df_path)
    
    generated_smiles_df_path = sys.argv[1]
    
    filterd_df = filtering(generated_smiles_df_path)
    filterd_df.to_csv("filtered.csv", index=False)
        
        
        
        
        
        
        
            
        