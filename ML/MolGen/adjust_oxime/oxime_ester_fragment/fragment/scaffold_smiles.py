from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def mol2img(mol, img_save_path):
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(img_save_path)


def getScaffoldSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smi = Chem.MolToSmiles(scaffold_mol)

    # remove side chain and get fragment scaffold
    scaffold_frag = Chem.ReplaceSidechains(mol, scaffold_mol)
    scaffold_frag_smi = Chem.MolToSmiles(scaffold_frag)
    return scaffold_smi, scaffold_frag_smi

if __name__ == '__main__':
    smiles = "CC(=O)O/N=C(\C(=O)C)C(=O)c(c1)ccc(c12)c3c(cccc3)C2(CCCC)CCCC"
    scaffold_smi, scaffold_frag_smi = getScaffoldSmiles(smiles)
    scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
    scaffold_frag_mol = Chem.MolFromSmiles(scaffold_frag_smi)
    mol2img(scaffold_mol, 'scaffold.png')
    mol2img(scaffold_frag_mol, 'scaffold_frag.png')
    
