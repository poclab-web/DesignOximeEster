from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from setting import *


def get_dataloader(batch_size, x, y, smiles, y_column, normalize=False):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=setting['TEST_SIZE'], random_state=int(setting['SEED']))
    smiles_train, smiles_test = train_test_split(smiles, test_size=setting['TEST_SIZE'], random_state=int(setting['SEED']))

    if normalize:
        y_mean, y_std = np.mean(y_train), np.std(y_train)
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
        print('平均:{}'.format(y_mean))
        print('分散:{}'.format(y_std))

    # save data
    df_train = pd.DataFrame(data=x_train)
    df_train.insert(0, 'Smiles', smiles_train)
    df_train[y_column] = y_train * y_std + y_mean
    df_test = pd.DataFrame(data=x_test)
    df_test.insert(0, 'Smiles', smiles_test)
    df_test[y_column] = y_test * y_std + y_mean

    df_train.to_csv('./cache/train_{}.csv'.format(y_column))
    df_test.to_csv('./cache/test_{}.csv'.format(y_column))

    transform = transforms.Compose([])

    dataset_train = MyDataSet(
        x_train,
        y_train,
        transforms=transform,
    )

    dataset_test = MyDataSet(
        x_test,
        y_test,
        transforms=transform,
    )

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=len(x_test),
        shuffle=True
    )

    return dataloader_train, dataloader_test


class MyDataSet(Dataset):
    def __init__(self, x, y, transforms):
        super().__init__()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.x = transforms(self.Tensor(x))
        self.y = transforms(self.Tensor(y))

        self.len = len(x)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return self.len