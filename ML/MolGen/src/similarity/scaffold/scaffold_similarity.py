import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


class ScaffoldSimilarityData:
    def __init__(self, smiles:str, scaffold_smi:str, similarity:float):
        self.smiles = smiles
        self.scaffold_smi = scaffold_smi
        self.similarity = similarity


class ScaffoldSimilarity:
    def __init__(self, base_smiles:list, radius:int=2, n_bits:int=2048):
        self.base_smiles = base_smiles
        self.radius = radius
        self.n_bits = n_bits
        self.base_mol_fp_list = self._smiles_to_fp(base_smiles, to_scaffold=True)
        
        print('all smiles are converted to mols and fingerprints')
    
    
    def _smiles_to_fp(self, smiles_list:list, to_scaffold=True):
        mols = [self._to_mol(s) for s in smiles_list]
        if to_scaffold:
            mols = [self._mol_to_scaffold(m) for m in mols]
        return [self._mol_to_fp(m) for m in mols]
    
    
    def _to_mol(self, smiles:str):
        return Chem.MolFromSmiles(smiles)
    
    def _mol_to_fp(self, mol):
        return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.n_bits)
    
    def _mol_to_scaffold(self, mol):
        return MurckoScaffold.GetScaffoldForMol(mol)
        
    
    def calculate_smimilarity(self, smiles:str):
        mol = self._to_mol(smiles)
        mol_scaffold = self._mol_to_scaffold(mol)
        mol_fp = self._mol_to_fp(mol_scaffold)
        
        similarity = DataStructs.BulkTanimotoSimilarity(mol_fp, self.base_mol_fp_list)
        max_similarity = max(similarity)
        
        return max_similarity, mol_scaffold
    
    def __call__(self, smiles_list:list):
        
        for smi in smiles_list:
            print('smi', smi)
            similarity, mol_scaffold = self.calculate_smimilarity(smi)
            smiles_scaffold = Chem.MolToSmiles(mol_scaffold)
            yield ScaffoldSimilarityData(smi, smiles_scaffold, similarity)


if __name__ == '__main__':
    import pandas as pd
    from rdkit.Chem import Draw
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    save_dir = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/src/similarity/scaffold/test"
    os.makedirs(save_dir, exist_ok=True)
    
    csv_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/data/results/generated_smiles/test.csv"
    df = pd.read_csv(csv_path)
    
    base_smiles_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/data/mydata/oxime_ester.smi"
    base_smiles = open(base_smiles_path).read().splitlines()
    
    
    scaffold_similarity = ScaffoldSimilarity(base_smiles)
    
    smiles_list = df["Smiles"].tolist()
    for idx, data in enumerate(scaffold_similarity(smiles_list)):
        mol_scaffold = Chem.MolFromSmiles(data.scaffold_smi)
        mol = Chem.MolFromSmiles(data.smiles)
        img = Draw.MolsToGridImage([mol, mol_scaffold], molsPerRow=2, legends=[data.smiles, data.scaffold_smi], subImgSize=(300, 300))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        # テキストの位置を左上に変更
        text_position = (10, 10) # 左上
        draw.text(text_position, str(data.similarity), fill=(0, 0, 0), font=font)
        save_img_path = os.path.join(save_dir, f"{idx}.png")
        img.save(save_img_path)
        
        
        
            
            