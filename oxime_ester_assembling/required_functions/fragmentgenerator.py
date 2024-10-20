from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Recap
import random
import re
import sys, os
import logging
sys.path.append(os.path.dirname(__file__))

from constraints import BaseConstraints

logger = logging.getLogger('Molecule Synthesis')

class MyConstraints(BaseConstraints):
    # original constraints
    def __init__(self, smarts):
        super().__init__(smarts)

def frgmentGenerator(smiles, method='brics'):
    if method == 'brics':
        frags = brics(smiles)
    
    elif method == 'recap':
        frags = recap(smiles)
    
    else:
        raise ValueError('You have to choose fragment generating method from brics and recap')
    
    # adapt constraints
    if frags is not None:
        match_frags = []
        for frag in frags:
            contrainer = MyConstraints(frag)
            smarts = contrainer.adapt_constraints()
            if smarts is not None:
                match_frags.append(smarts)
    
    else:
        logger.warning('No Fragment in {} using {}'.format(smiles, method))
        return None
    
    return match_frags



def brics(smiles):
    # generate fragments
    mol = Chem.MolFromSmiles(smiles)
    frags = list(BRICS.BRICSDecompose(mol))
    
    if len(frags) == 1:
        return None
    
    # single label
    frags = list(map(singleFrag, frags))

    return frags

def recap(smiles):
    mol = Chem.MolFromSmiles(smiles)
    recap = Recap.RecapDecompose(mol)
    frags = list(recap.children.values())
    frags = [f.smiles for f in frags]

    return frags


def singleFrag(fragment):
    # extract label
    reg = '\[\d+\*\]'
    frag_labels = re.findall(reg, fragment)
    
    if len(frag_labels) > 1:
        # randomly select a label 
        selected_label = random.choice(frag_labels)

        # delete selected label from label
        frag_labels.remove(selected_label)
        
        # delete unselected labels from fragment
        for l in frag_labels:
            fragment = fragment.replace(l, '')
        
        fragment = fragment.replace(selected_label, '*')
    
    else:
        fragment = fragment.replace(frag_labels[0], '*')
    
    # delete () and []
    fragment = fragment.replace('()', '')
    fragment = fragment.replace('[]', '')
    
    return fragment

def allocateFragLabel(frag, label):
    label = '[{}*]'.format(label)
    frag = frag.replace('*', label)
    return frag


if __name__ == '__main__':
    frag = 'CC1Oc2ccc(Cl)cc2N(CC(O)CO)C1=O'
    
    print(frgmentGenerator(frag, method='brics'))

    print(frgmentGenerator(frag, method='recap'))

    
