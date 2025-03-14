from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import random
import traceback

# Libraries
from Synthesizer import SynthesizerBase
from required_functions.reaction import react2fragments
from required_functions.fragmentgenerator import allocateFragLabel

#logger = logging.getLogger('Molecule Synthesis').getChild('assembly')
logger = logging.getLogger('Molecule Synthesis')

def assembly(filename:str, base_smarts:str, trg_smiles:str, method='recap'):
    """ 

    Args:
        filename: where to save csv
        base_smarts: base_smarts
        trg_smiles: 
    
    Returns:
        generated: list
    """

    # write file
    file_path = './result/{}.csv'.format(filename)

    logger.debug('file saved to {}'.format(file_path))

    try:
        syn = MySynthesizer(base_smarts, file_path, method)
        generated_smiles = syn.synthesis(trg_smiles)
        if generated_smiles is None:
            logger.error('failed to generate smiles!!')
            logger.error('base_smarts::: {}'.format(base_smarts))
            logger.error('trg_smiles::: {}'.format(trg_smiles))
            logger.error('method::: {}'.format(method))

    except Exception as e:
        generated_smiles = None
        logger.error(traceback.format_exc())
    
    return generated_smiles

class MySynthesizer(SynthesizerBase):
    def __init__(self, base_smarts, file_path, method):
        super().__init__(base_smarts, file_path, method)

    def synthesis(self, trg_smiles):
        # assign label index
        frag_num = self.smiles.count('*')

        logger.debug('frag_num:::'.format(frag_num))

        # replace "*" to label index from left 
        for i in range(frag_num):
            labelindex = str('{}00'.format(i + 1))
            labelindex = '[{}|]'.format(labelindex)
            self.smiles = self.smiles.replace('*', labelindex, 1)
        self.smiles = self.smiles.replace('|', '*')
        
        # fragment generator
        frags = self.fragment_generator(trg_smiles)

        if frags != [] and frags is not None:
            logger.info('fond fragments!!! number of fragment::: {}'.format(len(frags)))
            generated = []
            if len(frags) > frag_num:
                roop_num = len(frags) // frag_num
                logger.info('looping:::{}'.format(roop_num))

                for i in range(roop_num):
                    input_frags, frags = frags[:frag_num], frags[frag_num:]
                    smi = self.reactor(
                        base_smarts=self.smiles,
                        frags=input_frags,
                        rerturn_as_smiles=True
                    )
                    generated.append(smi)
            
            else:
                logger.info('not enough fragments')
                while len(frags) == frag_num - 1:
                    add = random.choice(frags)
                    frags.append(add)
                smi = self.reactor(
                    base_smarts=self.smiles,
                    frags=frags,
                    rerturn_as_smiles=True
                )
                generated.append(smi)
        
        
        else:
            return None
        
        logger.debug('generated:::{}'.format(generated))
        # saving step
        for smi in generated:
            self.save(smi)
        
        return generated


    def reactor(self, base_smarts, frags, rerturn_as_smiles=False):
        mol = Chem.MolFromSmarts(base_smarts)
        for idx, R in enumerate(frags):
            # addpt R
            logger.debug('R{} will be adapt: {}'.format(idx + 1, R))
            frag_id = str('{}00'.format(idx+1))
            # assign label to fragment
            R = allocateFragLabel(R, frag_id)
            R_mol = Chem.MolFromSmarts(R)
            mol = react2fragments(R_mol, mol, frag_id=frag_id)
        
        logger.debug('Finnaly Coverted:::{}'.format(Chem.MolToSmiles(mol)))

        if rerturn_as_smiles:
            return Chem.MolToSmiles(mol)

        return mol

if __name__ == '__main__':
    base_smarts = 'COC(=O)N=C(*)*'
    filename = 'test'
    trg_smiles = 'Cc1nc(C2CCCCC2)c(-c2ccc(S(N)(=O)=O)c(F)c2)o1'

    generated = assembly(filename, base_smarts, trg_smiles, method='brics')

    print(generated)