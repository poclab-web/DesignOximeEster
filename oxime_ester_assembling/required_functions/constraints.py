import sys, os
sys.path.append(os.path.dirname(__file__))

from rdkit import Chem
from molecule import getAtomNeighbors, getAtomIndex

import logging

logger = logging.getLogger('Molecule Synthesis')

class BaseConstraints:
    def __init__(self, smarts):
        logger.info('check:::{}'.format(smarts))
        self.smarts = smarts
        self.fragment_neighbor = 'C'
        self.use = True

    def adapt_constraints(self):
        self.smarts = self.numerical()
        self.smarts = self.filter()

        logger.debug('constraints result::: {}'.format('match' if self.smarts is not None else 'mismatch'))

        return self.smarts
    
    def numerical(self):
        return self.smarts
        
    
    def filter(self):
        
        if '*' not in self.smarts:
            logger.warning('{} is not fragment!!'.format(self.smarts))
            return None

        
        if self.fragment_neighbor:
            mol = Chem.MolFromSmarts(self.smarts)
            idxs = getAtomIndex(mol, '*')
            if idxs != []:
                atoms = getAtomNeighbors(mol, idxs[0], '*')
                symbol = atoms[0].GetSymbol()

                if symbol != self.fragment_neighbor:
                    self.use = False
                    return None
                
                else:
                    return self.smarts
            
            else:
                return None

if __name__ == '__main__':
    smarts = '*n1ccnc1'
    const = BaseConstraints(smarts)
    result = const.adapt_constraints()
    print(result)
    







    