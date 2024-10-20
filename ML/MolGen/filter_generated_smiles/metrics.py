import rdkit
from rdkit import Chem
import chem_utils.mol as mol_utils
import networkx as nx
from collections import Counter
import numpy as np
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor


def abtmpnn_predict(model, smiles: list, max_batch_size=100, num_threads=4):
    predict_cols = model.args.target_columns

    # separate smiles into batches
    smiles_batches = []
    for i in range(0, len(smiles), max_batch_size):
        smiles_batches.append(smiles[i:i+max_batch_size])

    # predict using multiple threads
    preds = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(model.predict, batch) for batch in smiles_batches]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            preds.append(future.result())

    # concat
    preds = np.concatenate(preds, axis=0)

    if preds.shape[0] == len(predict_cols):
        raise ValueError('preds.shape[0] must be equal to len(predict_cols).')

    pred_dict = {}
    for i, col in enumerate(predict_cols):
        pred_dict[col] = preds[:, i]

    return pred_dict


def get_max_fused_ring_size(smiles:str):
    mol = mol_utils.smiles2mol(smiles)

    try:
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
    except:
        print('error in get_max_fused_ring_size', smiles)
        ring_counts = [0]
    
    if len(ring_counts) == 0:
        return 0
    else:
        return max(ring_counts)
    
    
def symmetry(smiles:str):
    mol = mol_utils.smiles2mol(smiles)
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

def max_ring_size(smiles:str):
    mol = mol_utils.smiles2mol(smiles)
    # Find all rings in the molecule
    ssr = Chem.GetSymmSSSR(mol)
    if len(ssr) == 0:
        return 0
    # Calculate the size of the largest ring
    largest_ring_size = max(len(ring) for ring in ssr)

    return largest_ring_size


def num_oxime_activation(smiles:str):
    # Find all oxime groups
    oximes_num = mol_utils.count_oxime_activation(smiles)
    return oximes_num