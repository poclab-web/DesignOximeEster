from rdkit import Chem

def getAtomIndex(mol, symbol='*'):
    idxs = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol()  == symbol:
            idxs.append(atom.GetIdx())
    return idxs

def getAtomNeighbors(mol, atom_index, symbol='*'):
    neighbors = mol.GetAtomWithIdx(atom_index).GetNeighbors()
    return neighbors


if __name__ == '__main__':
    mol = Chem.MolFromSmarts('*OCc1cccc(CO)n1')
    index = getAtomIndex(mol)[0]

    print(getAtomNeighbors(mol, index))
    
