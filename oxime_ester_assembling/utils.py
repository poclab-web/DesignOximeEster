from rdkit import Chem
import rdkit
from rdkit.Chem import Draw
import re

def isIncludeSingleAtom(mol:rdkit.Chem.rdchem.Mol , atom_symbol:str):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == atom_symbol:
            continue
        else:
            return True
    return False

def changeFragExpr(frag_smi: str):
    frag_express = "[123*]"
    new_smiles = re.sub(r"\[(\d+)\*\]", frag_express, frag_smi)
    return new_smiles

def fragsToList(frag_smi: str):
    frags = frag_smi.split('.')
    frags = [Chem.MolFromSmiles(changeFragExpr(x)) for x in frags]
    return frags

def mol2img(mol, img_save_path):
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(img_save_path)

if __name__ == '__main__':
    frag_smi = "[1*]C(=O)/C(=N/OC(C)=O)C(C)=O.[2*]CCCC.[3*]CCCC"
    frags = fragsToList(frag_smi)
    print(frags)