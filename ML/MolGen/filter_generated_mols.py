import pandas as pd
import sys, os
abtmpnn_dir = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN"
sys.path.append(abtmpnn_dir)
from ABTMPNN import ABTMPNN
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm
from src.similarity.scaffold import ScaffoldSimilarity

class Filter:
    def __init__(self, original_smiles_path):
        
        original_smiles = open(original_smiles_path, "r").readlines()
        
        CHECKPOINT_PATH = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/fold_1/model_0/model.pt"
        ARGS_PATH = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/args.json"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.predic_model = ABTMPNN(CHECKPOINT_PATH, ARGS_PATH, DEVICE)
        self.target_t1, self.target_s1, self.target_s1_osc = 2.7, 2.5, 1.0
        self.scaffold_smilarity = ScaffoldSimilarity(original_smiles, radius=2, n_bits=2048)
    
    def split_smiles_to_batches(self, smiles, batch_size):
        return [smiles[i:i + batch_size] for i in range(0, len(smiles), batch_size)]
    
    def predict(self, smiles):
        smiles_batches = self.split_smiles_to_batches(smiles, 256)
        
        for smiles_batch in tqdm(smiles_batches, total=len(smiles_batches)):
            T1, S1, S1_osc = [], [], []
            try:
                pred_batch = self.predic_model.predict(smiles_batch)
                for pred in pred_batch:
                    T1.append(pred[0])
                    S1.append(pred[1])
                    S1_osc.append(pred[2])
                yield T1, S1, S1_osc, smiles_batch
            except:
                continue
            
    
    
    def tanimoto_similarity(self, smiles1):
        mol1 = Chem.MolFromSmiles(smiles1)
        
        mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        
        tanimoto_similarity = DataStructs.BulkTanimotoSimilarity(mol1_fp, self.original_fp)
        
        max_tanimoto_similarity = max(tanimoto_similarity)
        
        return max_tanimoto_similarity
    
    def remain_valid_smiles(self, smiles_list):
        valid_smiles = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_smiles.append(smi)
                else:
                    print('invalid smiles: ', smi)
            except:
                print('invalid smiles: ', smi)
                pass
        return valid_smiles
    
    
    def filtering(self, df):
        df = df[(df["T1"] > self.target_t1) & (df["S1"] > self.target_s1) & (df["S1_osc"] > self.target_s1_osc) & (df["Tanimoto_similarity"] < 0.5)]
        
        return df
    
    def __call__(self, generated_smiles_df_path, save_df_path):
        df = pd.DataFrame()
        smiles = pd.read_csv(generated_smiles_df_path)["Smiles"].tolist()
        smiles = self.remain_valid_smiles(smiles)
        predict_generator = self.predict(smiles)
        
        for (T1, S1, S1_osc, smiles_list) in predict_generator:
            # print("smiles_list ", smiles_list[0:10])
            similarity_generator = self.scaffold_smilarity(smiles_list)
            for i, similarity_data in enumerate(similarity_generator):
                similarity = similarity_data.similarity
                smiles_scaffold = similarity_data.scaffold_smi
                t1 = T1[i]
                s1 = S1[i]
                s1_osc = S1_osc[i]
                # print("t1, s1, s1_osc, similarity, smiles_scaffold", t1, s1, s1_osc, similarity, smiles_scaffold)
                df_add = pd.DataFrame({"Smiles": similarity_data.smiles, "T1": [t1], "S1": [s1], "S1_osc": [s1_osc], "similarity": similarity, "Scaffold": smiles_scaffold})
                df = pd.concat([df, df_add])
            df.to_csv(save_df_path, index=False)
                
        
if __name__ == "__main__":
    original_df_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/data/mydata/oxime_ester.smi"
    filtering = Filter(original_df_path)
    
    generated_smiles_df_path = sys.argv[1]
    output_path = sys.argv[2]
    
    filtering(generated_smiles_df_path, output_path)
        
        
        
        
        
        
        
            
        