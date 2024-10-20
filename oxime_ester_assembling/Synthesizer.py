import logging
from required_functions.fragmentgenerator import frgmentGenerator

logger = logging.getLogger('brics molecule Synthesis')

class SynthesizerBase:
    def __init__(self, base_smarts, file_path, method):
        self.smiles = base_smarts
        self.file_path = file_path
        self.method = method
    
    def save(self, smiles_gen):
        # save result
        with open(self.file_path, 'a') as f:
            print('{},{},{}\n'.format(self.smiles, smiles_gen, self.method), file=f, end='')
        f.close()
    
    def fragment_generator(self, trg_smiles):
        # generate frags from selected method
        frags = frgmentGenerator(trg_smiles, method=self.method)
        return frags
        