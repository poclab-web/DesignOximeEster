import pandas as pd

import sys, os
abtmpnn_dir = os.path.join(os.path.dirname(__file__), '../../ABT-MPNN')
sys.path.append(abtmpnn_dir)
from ABTMPNN import ABTMPNN


# abtmpnn
def predict(abtmpbnn_model, smiles:str):
    try:
        vals = abtmpbnn_model.predict([smiles])
        return vals
    except:
        import traceback
        print(traceback.print_exc())
        return None


if __name__ == "__main__":
    import numpy as np
    abtmpnn_checkpoint_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/fold_1/model_0/model.pt"
    abtmpnn_args_path = "/mnt/ssd2/Chem/photopolymerization_initiator/ML/ABT-MPNN/log/oxime_and_other_transed/args.json"
    model = ABTMPNN(abtmpnn_checkpoint_path, abtmpnn_args_path, 'cuda')
    
    df = pd.read_csv("/mnt/ssd2/Chem/photopolymerization_initiator/ML/MolGen/filter_generated_smiles/oxime_filter.csv")

    print('remove incalculable smiles now length {}'.format(len(df)))
    df['is_able_to_calculate'] = df['Smiles'].map(model.is_able_to_predict)
    df = df[df['is_able_to_calculate'] == True]
    print('remove incalculable smiles now length {}'.format(len(df)))

    smiles_list = df['Smiles'].values.tolist()

    predict_cols = model.args.target_columns

    val_dict = {}

    for col in predict_cols:
        val_dict[col] = []
    
    values = model.predict(smiles_list)
    values = np.array(values)

    print(values.shape)

    for i, col in enumerate(predict_cols):
        val_dict[col] = values[:, i]
    
    #add to df
    for col in predict_cols:
        df[col] = val_dict[col]
        
    
    df.to_csv("./oxime_pred_vals.csv")
        
        


