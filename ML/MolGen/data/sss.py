from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import networkx as nx
import rdkit
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from adjust_oxime.adjust_oxime import AdjustOxime

import pandas as pd


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

    if len(ring_counts) == 0:
        return 0
    return max(ring_counts)

def mol2smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def oxime_num(mol):
    oxime_activation_smiles = "C=NOC(=O)"
    oxime_activation_mol = Chem.MolFromSmiles(oxime_activation_smiles)
    oxime_num = len(mol.GetSubstructMatches(oxime_activation_mol))
    return oxime_num



def replace_bad_word(text):
    text = text.replace('\\', 'W')
    text = text.replace('/', 'Q')
    return text


def refine_smiles(text):
    text = text.replace('\\', 'W')
    text = text.replace('/', 'Q')
    return text

def mol2img(mol, img_save_path, text=None):
    # 分子の画像を生成
    img = Draw.MolToImage(mol, size=(300, 300))
    draw = ImageDraw.Draw(img)

    if text is not None:
        font = ImageFont.load_default()
        # テキストの位置を左上に変更
        text_position = (10, 10) # 左上
        draw.text(text_position, text, fill=(0, 0, 0), font=font)

    # 画像をファイルに保存
    img.save(img_save_path)


def load_smiles_from_csv(csv_path, smiles_col='Smiles'):
    return pd.read_csv(csv_path)[smiles_col].values.tolist()


def load_smiles_from_smi(smi_path):
    with open(smi_path) as f:
        smiles_list = f.readlines()
    smiles_list = [smi.strip() for smi in smiles_list]
    return smiles_list


def save_data_frame(data_dict, save_path):
    df = pd.DataFrame(data_dict)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    smiles_list = load_smiles_from_csv(csv_path)

    all_smiles_length = len(smiles_list)
    valid_smi_num = 0
    for smi in tqdm(smiles_list):
        mol = mol2smiles(smi)

        if mol is not None and oxime_num(mol) > 1 and get_fused_ring_counts(mol) > 2:
            smi = Chem.MolToSmiles(mol)
            valid_smi_num += 1
            smiles_file_name = replace_bad_word(smi)
            save_path = os.path.join(save_dir, smiles_file_name + '.png')
            mol2img(mol, save_path, text=smi)
    
    save_csv_path = os.path.join(save_dir, 'valid_smiles.csv')
    save_data_frame({'Smiles': smiles_list}, save_csv_path)
    print('{} valid / {} all smiles'.format(valid_smi_num, all_smiles_length))
