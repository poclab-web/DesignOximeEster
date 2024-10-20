from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

class OximeEsterAdjuster:
    def __init__(self, molecule_smiles):
        self.molecule = Chem.MolFromSmiles(molecule_smiles)
        self.target_substructure = Chem.MolFromSmarts('C=NOC(=O)')

    def get_substructure(self, atom_index):
        # Extract the substructure bonded to the given atom, excluding the target substructure
        neighbors = [atom.GetIdx() for atom in self.molecule.GetAtomWithIdx(atom_index).GetNeighbors()]
        sub_mol = Chem.PathToSubmol(self.molecule, neighbors)
        print(Chem.MolToSmiles(sub_mol))
        smarts = Chem.MolToSmarts(sub_mol)
        if sub_mol.HasSubstructMatch(self.target_substructure):
            return None
        else:
            return smarts
    
    def oxime_curbon_idx(self, oxime_ester_sub_idxs, neibor_symbol='N'):
        # get NC OC(=O) carbon index
        assert neibor_symbol in ['N', 'O'], 'neibor_symbol must be N or O'
        for idx in oxime_ester_sub_idxs:
            if self.molecule.GetAtomWithIdx(idx).GetSymbol() == 'C':
                neibors_atoms = self.molecule.GetAtomWithIdx(idx).GetNeighbors()
                for neibor in neibors_atoms:
                    if neibor.GetSymbol() == neibor_symbol:
                        return idx
        return None
    

    def _oco_connected_substruct(self, oco_carbon_index_in_molecule):
        # Extract the substructure bonded to the given atom, excluding the target substructure
        neighbors = self.molecule.GetAtomWithIdx(oco_carbon_index_in_molecule).GetNeighbors()
        
        for neighbor in neighbors:
            if neighbor.GetSymbol() != 'O':
                neighbor_index = neighbor.GetIdx()
                break
                        


    def analyze(self):
        # Find all instances of the target substructure
        matches = self.molecule.GetSubstructMatches(self.target_substructure)

        # Variables to store the found substructures
        oc_substructures = []
        nc_substructures = []

        for match_idxs in matches:
            # OC(=O) carbon and N=C carbon indices
            oco_carbon_idx = self.oxime_curbon_idx(match_idxs, neibor_symbol='O')
            nc_carbon_idx = self.oxime_curbon_idx(match_idxs, neibor_symbol='N')

            assert oco_carbon_idx is not None and self.molecule.GetAtomWithIdx(oco_carbon_idx).GetSymbol() == 'C', 'oco_carbon_idx is not None and self.molecule.GetAtomWithIdx(oco_carbon_idx).GetSymbol() == C'
            assert nc_carbon_idx is not None and self.molecule.GetAtomWithIdx(nc_carbon_idx).GetSymbol() == 'C', 'nc_carbon_idx is not None and self.molecule.GetAtomWithIdx(nc_carbon_idx).GetSymbol() == C'

            # Analyze OC(=O) substructure
            oc_substructure = self.get_substructure(oc_carbon_idx)
            oc_substructures.append(oc_substructure)
            print(oc_substructure)

            # Analyze N=C substructure
            nc_neighbors = self.molecule.GetAtomWithIdx(nc_carbon_idx).GetNeighbors()
            print(nc_neighbors)
            nc_substructures.extend([self.get_substructure(atom.GetIdx()) for atom in nc_neighbors])

        # Apply the selection criteria for N=C substructures
        if nc_substructures.count(None) != len(nc_substructures):
            nc_substructures = [sub for sub in nc_substructures if sub]
            if nc_substructures:
                weights = [Descriptors.MolWt(Chem.MolFromSmiles(sub)) for sub in nc_substructures]
                selected_nc_substructure = nc_substructures[weights.index(min(weights))]
            else:
                selected_nc_substructure = None
        else:
            selected_nc_substructure = None

        return oc_substructures, selected_nc_substructure

    def substitute_structure(self, substructure_smarts, replacement_smarts):
        # Substitute the identified substructure with another structure
        substructure = Chem.MolFromSmarts(substructure_smarts)
        replacement = Chem.MolFromSmiles(replacement_smarts)
        return AllChem.ReplaceSubstructs(self.molecule, substructure, replacement)


if __name__ == "__main__":
    # Example usage
    molecule_smiles = "CC(=O)O/N=C(\C)c1ccc(NC(=O)/C=C/c2ccccc2)cc1"
    replacement_smiles = 'C'
    analyzer = OximeEsterAdjuster(molecule_smiles)
    oc_substructures, nc_substructure = analyzer.analyze()

    print(oc_substructures, nc_substructure)

    # Replace identified substructures with the replacement structure
    for oc_substructure in oc_substructures:
        if oc_substructure:
            analyzer.substitute_structure(oc_substructure, replacement_smiles)

    if nc_substructure:
        analyzer.substitute_structure(nc_substructure, replacement_smiles)
