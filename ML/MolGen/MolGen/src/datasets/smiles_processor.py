from rdkit import Chem
import numpy as np

def randomize_smiles(smiles):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)


if __name__ == '__main__':
    import sys
    input_smiles = sys.argv[1]
    print(input_smiles)
    print(randomize_smiles(input_smiles))
    