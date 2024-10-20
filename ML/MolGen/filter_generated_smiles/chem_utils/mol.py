from rdkit import Chem

def smiles2mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None


def smiles2mol2smiles(smiles):
    mol = smiles2mol(smiles)
    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol)
    
def count_oxime_activation(smiles):
    mol = smiles2mol(smiles)
    return len(mol.GetSubstructMatches(Chem.MolFromSmiles("C=NOC(=O)")))