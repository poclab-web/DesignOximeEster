from rdkit import Chem
from rdkit.Chem import FunctionalGroups
from .scaffold_smiles import getScaffoldSmiles


class FragmentSmiles:
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.fgs = FunctionalGroups.BuildFuncGroupHierarchy()
        
    
    def getScaffold(self):
        scaffold_smiles, scaffold_smiles_frag = getScaffoldSmiles(self.smiles)
        return scaffold_smiles, scaffold_smiles_frag

    
    def getFunctionalGroups(self):
        match_fgs = []
        for fg in self.fgs:
            patt = fg.pattern
            res = self.mol.HasSubstructMatch(patt)
            if res:
                patt_smiles = Chem.MolToSmiles(patt)
                match_fgs.append(patt_smiles)
        return match_fgs

if __name__ == '__main__':
    smi = "CC(=O)O/N=C(\C(=O)C)C(=O)c(c1)ccc(c12)c3c(cccc3)C2(CCCC)CCCC"
    fragment = FragmentSmiles(smi)
    print(fragment.getFunctionalGroups())
    print(fragment.getScaffold())
    
