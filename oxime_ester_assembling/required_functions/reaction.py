from rdkit import Chem
from rdkit.Chem import AllChem

def react2fragments(frag1, frag2, frag_id):
    # 2分子を反応させる
    rxn_AB = AllChem.ReactionFromSmarts("[*:1][{}*].[*:2][{}*]>>[*:1][*:2]".format(frag_id, frag_id))
    return rxn_AB.RunReactants([frag1, frag2])[0][0]