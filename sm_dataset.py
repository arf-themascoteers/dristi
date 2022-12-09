import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class SM_Dataset(Dataset):
    def __init__(self, dstype="all"):
        sm = pd.read_csv("sm.csv").to_numpy()
        self.X = sm[:, 1:]
        self.y = sm[:, 0:1]

        scaler = MinMaxScaler()
        self.y = scaler.fit_transform(self.y)
        self.y = self.y.squeeze()


        self.X, X_test, self.y, y_test = train_test_split(self.X, self.y, random_state=1)

        if dstype == "all":
            self.X = np.concatenate((self.X, X_test), axis=0)
            self.y = np.concatenate((self.y, y_test), axis=0)

        elif dstype == "test":
            self.X = X_test
            self.y = y_test

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]