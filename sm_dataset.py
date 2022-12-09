import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from PyEMD import EMD
from sklearn import decomposition
from sklearn.utils import shuffle


class SM_Dataset(Dataset):
    def __init__(self, data_type="vb"):
        sm = shuffle(pd.read_csv("sm.csv").to_numpy())
        self.X = sm[:, 1:]
        self.y = sm[:, 0:1]

        scaler = MinMaxScaler()
        self.y = scaler.fit_transform(self.y)
        self.y = self.y.squeeze()

        if data_type == "emd":
            imfs = self._get_emd(self.X)
            self.X = np.concatenate((self.X, imfs), axis=1)

        if data_type == "pca":
            pca = decomposition.PCA(n_components=7)
            components = pca.fit_transform(self.X)
            self.X = components

        if data_type == "pca_emd":
            imfs = self._get_emd(self.X)
            self.X = np.concatenate((self.X, imfs), axis=1)
            pca = decomposition.PCA(n_components=7)
            components = pca.fit_transform(self.X)
            self.X = components

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def _get_emd(self, X):
        emd = EMD()
        IMFs = np.zeros((X.shape[0], X.shape[1]))
        for index, row in enumerate(X):
            x = emd(row)
            variations = np.var(x, axis=1)
            IMFs[index] = x[0]

        IMFs = IMFs.reshape(IMFs.shape[0], -1)
        return IMFs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    # ds = SM_Dataset(data_type="vb")
    # print(ds.X.shape)
    #
    # ds = SM_Dataset(data_type="emd")
    # print(ds.X.shape)
    #
    # ds = SM_Dataset(data_type="pca")
    # print(ds.X.shape)

    ds = SM_Dataset(data_type="pca_emd")
    print(ds.X.shape)