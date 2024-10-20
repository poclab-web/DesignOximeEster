from rdkit import Chem
from rdkit.Chem import AllChem

from fragment import FragmentSmiles
from utils import (
    isIncludeSingleAtom, changeFragExpr, fragsToList
)


class OximeEsterFragment(FragmentSmiles):
    def __init__(self, smiles):
        super().__init__(smiles)
        self.oxime_ester_base_struct = "C=NOC(=O)"
        self.oxime_ester_base_struct_mol = Chem.MolFromSmiles(self.oxime_ester_base_struct)
        self.oxime_ester_base_struct_smart = "C=NOC(=O)*"

    def getScaffold(self):
        scaffold_smiles, scaffold_smiles_frag = super().getScaffold()

        return scaffold_smiles, scaffold_smiles_frag
    
    def getOximeEsterBaseStruct(self):
        substruct_match = self.mol.GetSubstructMatch(Chem.MolFromSmarts(self.oxime_ester_base_struct_smart))
        if substruct_match:
            submol = Chem.PathToSubmol(self.mol, substruct_match)
            return Chem.MolToSmiles(submol)
        else:
            return None

    
    def alkylChain(self):
        # get oxime ester base struct
        scaffold_smiles, _ = self.getScaffold()
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

        # replace base struct and get alkyl chain
        side_chaines_mol = AllChem.ReplaceCore(self.mol, scaffold_mol)
        side_chaines_smi = Chem.MolToSmiles(side_chaines_mol)
        side_chaines_mols = fragsToList(side_chaines_smi)
        # side_chaines = AllChem.DDeleteSubstructseleteSubstructs(self.mol, scaffold_mol)

        # remove oxime ester base struct
        alkyl_chaines = [x for x in side_chaines_mols if not x.HasSubstructMatch(self.oxime_ester_base_struct_mol) and isIncludeSingleAtom(x, 'C')]

        # to smiles
        alkyl_chaines = [Chem.MolToSmiles(x) for x in alkyl_chaines]

        return list(set(alkyl_chaines))
    
if __name__ == '__main__':
    smi = "CC(=O)O/N=C(\C(=O)C)C(=O)c(c1)ccc(c12)c3c(cccc3)C2(CCCC)CCCC"
    oxime_ester_frag = OximeEsterFragment(smi)
    alkyl_chaines = oxime_ester_frag.alkylChain()
    print('base structs', oxime_ester_frag.getOximeEsterBaseStruct())
    print('scaffold', oxime_ester_frag.getScaffold())
    print(alkyl_chaines)
    