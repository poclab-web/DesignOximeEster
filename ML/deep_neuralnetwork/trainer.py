import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from networks.dense_network import DenseNetwork
from dataloader import get_dataloader


class Trainer:
    def __init__(self, opt):
        self.cuda = True if torch.cuda.is_available() else False
        self.device = torch.device(
            str("cuda:0") if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor

        self.opt = opt

        ### data processing ###
        df = pd.read_csv(self.opt.df_path, index_col=0)
        self.x = df.iloc[:, 1:-1].values
        self.y = df[self.opt.y_column].values
        smiles = df['Smiles'].values

        self.dataloder = get_dataloader(opt.batch_size, self.x, self.y, smiles, self.opt.y_column, normalize=self.opt.normalize)
        
        self.network = DenseNetwork(len(self.x[0])).to(self.device)

        #self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=opt.learning_rate)

    def train(self):
        data_train, data_test = self.dataloder.__iter__()
        data_train = iter(data_train).next()

        x = data_train[0]
        y_true = data_train[1]

        self.network.train()
        self.optimizer.zero_grad()
        y_pred = self.network(x)
        y_pred = torch.squeeze(y_pred) 
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.optimizer.step()

        #random_index = np.random.choice(range(len(y_true)))
        #print(y_true[random_index], y_pred[random_index])

        r2_train = r2_score(np.squeeze(y_true.to('cpu').detach().numpy().copy()), np.squeeze(y_pred.to('cpu').detach().numpy().copy()))

        # loss = torch.mean(loss)

        data_test = iter(data_test).next()

        x_test = data_test[0]
        y_test = data_test[1]
        y_pred = self.network(x_test)

        y_test = y_test.to('cpu').detach().numpy().copy()
        y_pred = y_pred.to('cpu').detach().numpy().copy()

        y_test = np.squeeze(y_test)
        y_pred = np.squeeze(y_pred)

        #random_index = np.random.choice(range(len(y_test)))
        #print(y_test[random_index], y_pred[random_index])

        r2_test = r2_score(y_test, y_pred)

        return loss, r2_train, r2_test
