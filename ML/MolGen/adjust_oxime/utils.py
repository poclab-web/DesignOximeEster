from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image, ImageDraw, ImageFont

def check_neibors(mol, atom_idx):
    neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
    return neighbors

def mol2smiles(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    except:
        return None

def replace_substructure(molecule, substructure, replacement):
    """
    Replaces a specific substructure in a molecule with another substructure.

    Parameters:
    - molecule (rdkit.Chem.Mol): The original molecule.
    - substructure (rdkit.Chem.Mol): The substructure to replace.
    - replacement (rdkit.Chem.Mol): The replacement substructure.

    Returns:
    - rdkit.Chem.Mol: The modified molecule.
    """
    # Find the substructure matches in the molecule
    matches = molecule.GetSubstructMatches(substructure)

    # If no matches found, return the original molecule
    if not matches:
        return molecule

    # Replace the first match (for simplicity)
    # Note: This can be modified to replace all matches or specific ones
    rms = AllChem.ReplaceSubstructs(molecule, substructure, replacement)
    
    # Return the first modified molecule
    return rms[0]


def replace_wildcard_with_substituent(base_smiles, substituent_smiles):
    """
    Replaces a wildcard '*' in a SMILES string with a specified substituent.

    Parameters:
    - base_smiles (str): The base SMILES string containing a wildcard '*'.
    - substituent_smiles (str): The SMILES string of the substituent to replace the wildcard.

    Returns:
    - str: The SMILES string of the modified molecule.
    """
    # Create a molecule from the base SMILES
    base_mol = Chem.MolFromSmiles(base_smiles)

    # Create a molecule from the substituent SMILES
    substituent_mol = Chem.MolFromSmiles(substituent_smiles)

    # Define the query for the wildcard
    query = Chem.MolFromSmarts('[*]')

    # Find the atom indices where the wildcard is located
    atom_indices = base_mol.GetSubstructMatches(query)

    # Replace the wildcard with the substituent
    mol = AllChem.ReplaceSubstructs(base_mol, query, substituent_mol)[-1]

    # Return the modified SMILES
    return Chem.MolToSmiles(mol)


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