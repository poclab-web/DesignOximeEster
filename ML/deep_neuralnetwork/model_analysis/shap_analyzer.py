import shap
from neural_network.networks.dense_network import BDE_estimator
import os
import torch
from dataframe.load import load_csv
from dotenv import load_dotenv
load_dotenv()

def getSHAP(model_path, df_path, max_display, data_length):
    MODEL_PATH = model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BDE_estimator(first_node=data_length).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    df = load_csv(df_path)

    x = torch.FloatTensor(df.iloc[:, 1:-1].values).to(device)
    de = shap.DeepExplainer(model, data=x)

    shap_vals = de.shap_values(x)
    print(shap_vals.shape)

    shap.summary_plot(shap_values=shap_vals, features=x, max_display=max_display)

    return shap_vals