import sys, os
import pandas as pd

from reward_fn import OximePropReward

reward = OximePropReward("OximePropReward", None)
csv_path = '/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/data/oxime_and_other.csv'
df = pd.read_csv((csv_path))
smiles_list = df['smiles'].values.tolist()[:10]
print(reward(smiles_list))
