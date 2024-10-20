import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdMolDescriptors, RDConfig

try:
    from .fragment import FragmentSmiles
    from .utils import (
        isIncludeSingleAtom, changeFragExpr, fragsToList)
except ImportError:
    from fragment import FragmentSmiles
    from utils import (
        isIncludeSingleAtom, changeFragExpr, fragsToList)


class OximeEsterFragment(FragmentSmiles):
    def __init__(self, smiles):
        super().__init__(smiles)
        self.oxime_ester_base_struct = "C=NOC(=O)"
        self.oxime_ester_base_struct_mol = Chem.MolFromSmiles(self.oxime_ester_base_struct)
        self.oxime_ester_base_struct_smart = "C=NOC(=O)*"
        self.oxime_ester_base_struct_smart_mol = Chem.MolFromSmiles(self.oxime_ester_base_struct_smart)

    def getScaffold(self):
        scaffold_smiles, scaffold_smiles_frag = super().getScaffold()
        return scaffold_smiles, scaffold_smiles_frag
    
    def getOximeEsterBaseStruct(self, return_as_mol=False):
        substruct_match = self.mol.GetSubstructMatch(Chem.MolFromSmarts(self.oxime_ester_base_struct_smart))
        if substruct_match:
            submol = Chem.PathToSubmol(self.mol, substruct_match)
            if return_as_mol:
                return submol
            return Chem.MolToSmiles(submol)
        else:
            return None
    
    def moleculeweight(self, mol):
        return rdMolDescriptors.CalcExactMolWt(mol)
    
    def get_side_chains(self):
        mol = Chem.AddHs(self.mol)
        Chem.SanitizeMol(mol)
        # Identify the active site and the bonds to break
        active_site_pattern = Chem.MolFromSmarts('C=NOC(=O)')
        active_site_matches = mol.GetSubstructMatches(active_site_pattern)
        
        if not active_site_matches:
            return None  # No active site found
        
        # Assume the first match is the one we're interested in
        active_site_match = active_site_matches[0]
        c_atom_idx = active_site_match[0]  # Carbon atom index
        
        # Find the bonds to break
        bonds_to_break = [bond_idx for bond_idx, bond in enumerate(mol.GetBonds())
                        if bond.GetBeginAtomIdx() == c_atom_idx or bond.GetEndAtomIdx() == c_atom_idx]
        
        # Break the bonds
        emol = Chem.EditableMol(mol)
        for bond_idx in sorted(bonds_to_break, reverse=True):  # Reverse sort to maintain valid indices
            bond = mol.GetBondWithIdx(bond_idx)
            emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        fragmented_mol = emol.GetMol()
        
        # Get the fragments
        frags = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)

        molwts = []
        for frag in frags:
            molwt = self.moleculeweight(frag)
            molwts.append(molwt)
        
        max_index = molwts.index(min(molwts))
        side_chain = frags[max_index]
        side_chain = Chem.RemoveHs(side_chain)
        return side_chain

    
    def alkylChain(self):
        # get oxime ester base struct
        scaffold_smiles, _ = self.getScaffold()
        scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)

        # replace base struct and get alkyl chain
        side_chaines_mol = AllChem.ReplaceCore(self.mol, self.oxime_ester_base_struct_mol)
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
    print('side chain', Chem.MolToSmiles(oxime_ester_frag.get_side_chains()))
    print(alkyl_chaines)
    