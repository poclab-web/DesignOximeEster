from abc import ABC, abstractmethod
from collections import OrderedDict

import math
import numpy as np
from typing import Callable, Optional, Union, List

import chemprop

from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy

import networkx as nx
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import qed
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
import rdkit
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
import torch
from tqdm.auto import tqdm
from vina import Vina
from collections import Counter


def get_reward_fn(reward_names: List[str], paths: List[str]=None, multipliers: List[str]=None, **kwargs):
    reward_fns = []
    print('reward_names', reward_names)

    multipliers = [multipliers[0] for i in range(len(reward_names))]
    for reward_name, path, mult in zip(reward_names, paths, multipliers):
        print('reward_name, path, mult', reward_name, path, mult)
        if reward_name == 'Anti Cancer':
            reward_fn = ChempropReward(path, reward_name, multiplier=eval(mult))

        elif reward_name == 'LIDI':
            reward_fn = ChempropReward(path, reward_name, multiplier=eval(mult))

        elif reward_name == 'Sim':
            reward_fn = SimilarityReward(path, reward_name, multiplier=eval(mult))
        
        elif reward_name == 'QED':
            reward_fn = QEDReward(reward_name, multiplier=eval(mult))

        elif reward_name == 'Docking':
            reward_fn = DockingReward(path, reward_name, multiplier=eval(mult))
        
        elif reward_name == 'Symmetry':
            reward_fn = SymmetryReward(reward_name, multiplier=eval(mult))
        
        elif reward_name == 'FusedRing':
            reward_fn = FusedRingReward(reward_name, multiplier=eval(mult))
        
        elif reward_name == 'NumAromaticHeterocycles':
            reward_fn = NumAromaticHeterocyclesReward(reward_name, multiplier=eval(mult))
        
        elif reward_name == "MoleculeWeight":
            reward_fn = MoleculeWeightReward(reward_name, multiplier=eval(mult))
        

        reward_fns.append(reward_fn)

    if len(reward_fns) == 1:
        print('single reward')
        return reward_fns[0]
    else:
        print('multi reward')
        return MultiReward(name='MultiReward', reward_fns=reward_fns)

class Reward(ABC):
    def __init__(self, name, multiplier:  Optional[Callable[[float], float]]=None, eval_: bool=False, **kwargs) -> None:
        self.name = name
        self.multiplier = multiplier
        self._eval = eval_
        
    @abstractmethod
    def __call__(self, smiles: str):
        raise NotImplementedError

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, val):
        self._eval = val

    def __str__(self,):
        return self.name

class MultiReward(Reward):
    def __init__(self, name, reward_fns) -> None:
        super().__init__(name=name)
        self.reward_fns = reward_fns

    def __call__(self, smiles):
        rewards = OrderedDict()
        for fn in self.reward_fns:
            reward = fn(smiles)
            rewards[str(fn)] =  reward

        if not self.eval:
            rewards = list(zip(*list(rewards.values())))
            rewards = [sum(rewards) for rewards in rewards]

        return rewards

    @Reward.eval.setter
    def eval(self, val):
        for fn in self.reward_fns:
            if hasattr(fn, '_eval'):
                fn.eval = val
        Reward.eval.fset(self, val)

class DockingReward(Reward):
    def __init__(self, receptor_path, name, multiplier=None, **kwargs):
        super().__init__(name=f"{receptor_path.split('/')[-1].split('.')[0]}_{name}", multiplier=multiplier, **kwargs)

        self.receptor_path = receptor_path
        protein = Chem.MolFromPDBFile(receptor_path[:-2]) #we remove the last 2 chars with will resutls in reading the PDB file
        pos = protein.GetConformer(0).GetPositions()
        self.center = (pos.max(0) + pos.min(0)) / 2
        print(f"Protein center is: {self.center}")

        self.vina = Vina(sf_name='vina', cpu=0, verbosity=1)

    def __call__(self, smiles: List[str]):
        if isinstance(smiles, str):
            print("Converting smiles to list")
            smiles = [smiles]

        rewards = [self.__dock(s) if Chem.MolFromSmiles(s) is not None else 0 for s in tqdm(smiles)]

        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards

    def __dock(self, smiles):
        try:
            # Create RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if mol.GetNumConformers() > 0:
                AllChem.MMFFOptimizeMolecule(mol)

            else:
                return 0

            # Prepare mol
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            for setup in mol_setups:
                pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)


            if not is_ok:
                return 0
            # with open(f"./data/proteins/{smiles}.pdbqt", 'w') as f:
            #    f.write(pdbqt_string)
        
            # Configure Vina
            self.vina.set_receptor(self.receptor_path)
            self.vina.set_ligand_from_string(pdbqt_string)

            # Define the search space (coordinates and dimensions)
            x, y, z = self.center
            self.vina.compute_vina_maps(center=[x, y, z], box_size=[30, 30, 30])

            # Run docking
            self.vina.dock(n_poses=5, exhaustiveness=32)

            score = self.vina.score()[0]

            return score

        except Exception:
            return 0


class SimilarityReward(Reward):
    def __init__(self, smiles, name, multiplier=None, **kwargs):
        super().__init__(name=name, multiplier=multiplier, **kwargs)

        self.name = name
        self.smiles = smiles
        self.fp = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles))

    def __call__(self, smiles: List[str]):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        filtered_mols = [mol for mol in mols if mol is not None]
        fps = [Chem.RDKFingerprint(mol) for mol in filtered_mols]
        sims = DataStructs.BulkTanimotoSimilarity(self.fp, fps)

        rewards = [sims[filtered_mols.index(mols[i])] if mols[i] is not None else 0 for i in range(len(mols))]

        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards

class ChempropReward(Reward):
    def __init__(self,
                predictor_path,
                name: str='Chemprop',
                multiplier: Optional[Callable[[float], float]]=None,
                **kwargs) -> None:
        super().__init__(name=name, multiplier=multiplier, **kwargs)
        arguments = [
            '--test_path', '/dev/null',
            '--preds_path', '/dev/null',
            '--checkpoint_dir', predictor_path,
            #'--features_generator', 'rdkit_2d_normalized',
            '--no_features_scaling',
        ]

        self.args = chemprop.args.PredictArgs().parse_args(arguments)

        self.model_objects = chemprop.train.load_model(args=self.args)

    def __call__(self, smiles: str) -> float:
        if isinstance(smiles, list):
            smiles = [[s] for s in smiles]
        else:
            smiles = [[smiles]]
        preds = []
        try:
            preds = chemprop.train.make_predictions(args=self.args, smiles=smiles, model_objects=self.model_objects)
            preds = [pred[0] if pred is not None and pred[0] != 'Invalid SMILES' else 0 for pred in preds]
        except TypeError:
            print('Faild to make predictions')
            for s in smiles:
                try:
                    pred = chemprop.train.make_predictions(args=self.args, smiles=[s], model_objects=self.model_objects)
                    preds.append(pred[0] if pred is not None and pred[0] != 'Invalid SMILES' else 0)
                except TypeError:
                    print(f'Bad SMILES: {s[0]}')
                    preds.append(0)

        if self.multiplier is not None and not self.eval:
            #print(preds[:5])
            preds = [self.multiplier(pred) for pred in preds]
            #print(preds[:5])

        return preds

class QEDReward(Reward):
    def __init__(self,
                 name,
                 multiplier=None,
                 **kwargs):

        super(QEDReward, self).__init__(name=name, multiplier=multiplier)

    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = [qed(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None else 0 for s in smiles]
        
        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards


class MoleculeWeightReward(Reward):
    def __init__(self,
                 name,
                 multiplier=None,
                 **kwargs):
        self.target = 500
        super(MoleculeWeightReward, self).__init__(name=name, multiplier=multiplier)

    def mol_weight(self, mol:rdkit.Chem.rdchem.Mol):
        mol_wt = Descriptors.MolWt(mol)
        reward = gaussian_reward(mol_wt, target=self.target, sigma=50)
        return reward

    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = [self.mol_weight(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None else 0 for s in smiles]
        
        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards

class SymmetryReward(Reward):
    def __init__(self,
                 name,
                 multiplier=None,
                 **kwargs):

        super(SymmetryReward, self).__init__(name=name, multiplier=multiplier)

    def symmetry(self, mol:rdkit.Chem.rdchem.Mol):
        # 分子中のすべての原子のランクを取得
        rank_atoms = rdkit.Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)
        # 各ランクの出現回数をカウント
        rank_count = Counter(rank_atoms)

        # 等価な原子ペアの数を計算
        equivalent_atom_pairs = sum(count // 2 for count in rank_count.values())

        # 対称性スコアを計算（全原子ペアの数に対する等価な原子ペアの割合）
        total_atom_pairs = len(rank_atoms) // 2
        symmetry_score = equivalent_atom_pairs / total_atom_pairs if total_atom_pairs > 0 else 0

        return symmetry_score

    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = [self.symmetry(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None else 0 for s in smiles]
        
        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards


def get_fused_ring_counts(mol:rdkit.Chem.rdchem.Mol):
    """
    Count the number of individual rings in each fused ring system of a molecule.

    :param mol: RDKit molecule object
    :return: A list where each element represents the number of rings in a fused ring system
    """
    # Get the smallest set of smallest rings (SSSR)
    sssr = Chem.GetSymmSSSR(mol)

    # Create a graph where each node represents a ring
    G = nx.Graph()

    # Convert each ring to a set of bond indices and add them as nodes
    ring_bonds = [set(ring) for ring in sssr]
    for i in range(len(ring_bonds)):
        G.add_node(i)

    # Add edges between rings that share bonds
    for i, ring1 in enumerate(ring_bonds):
        for j, ring2 in enumerate(ring_bonds):
            if i < j and not ring1.isdisjoint(ring2):
                G.add_edge(i, j)

    # Find connected components, each representing a fused system
    fused_systems = list(nx.connected_components(G))

    # Count the number of rings in each fused system
    ring_counts = [len(system) for system in fused_systems]

    return ring_counts

# Define the Gaussian reward function
def gaussian_reward(x, target=3, sigma=1):
    return np.exp(-((x - target)**2) / (2 * sigma**2))

class FusedRingReward(Reward):
    def __init__(self,
                 name,
                 multiplier=None,
                 **kwargs):
        self.target_fused_rings = 3
        super(FusedRingReward, self).__init__(name=name, multiplier=multiplier)

    def calculate_ring_reward(self, mol:rdkit.Chem.rdchem.Mol):
        fused_rings = get_fused_ring_counts(mol)

        if len(fused_rings) == 0:
            max_fused_rings = 0
        else:
            max_fused_rings = max(fused_rings)
        
        reward = gaussian_reward(max_fused_rings, target=self.target_fused_rings, sigma=1)
        # difference = abs(max_fused_rings - self.target_fused_rings)
        # k = 1.0
        # reward = 1 / (1 + math.exp(k * difference))

        return reward


    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = [self.calculate_ring_reward(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None else 0 for s in smiles]
        
        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards



class NumAromaticHeterocyclesReward(Reward):
    def __init__(self,
                 name,
                 multiplier=None,
                 **kwargs):
        self.target_num_ahc_rings = 2
        super(NumAromaticHeterocyclesReward, self).__init__(name=name, multiplier=multiplier)

    def calculate_HAC_reward(self, mol:rdkit.Chem.rdchem.Mol):
        num_hc = Descriptors.NumAromaticHeterocycles(mol)

        rewad = gaussian_reward(num_hc, target=3, sigma=1)
        return reward


    def __call__(self, smiles: str):
        if isinstance(smiles, str):
            smiles = [smiles]
        
        rewards = [self.calculate_HAC_reward(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None else 0 for s in smiles]
        
        if self.multiplier is not None and not self.eval:
            rewards = [self.multiplier(reward) for reward in rewards]

        return rewards

def main():
    # with QEDReward() as rfn:
    #     print(rfn("CCO"))

    # with QEDReward() as rfn:
    #     print(rfn("CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]=]"))
    
    with MoleculeWeightReward("molwt", "lamda x: x") as rfn:
        print(rfn("CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]=]"))

        
if __name__ == "__main__":
    main()
