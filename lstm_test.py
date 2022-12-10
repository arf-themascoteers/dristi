import torch
from sklearn.linear_model import LinearRegression
from sm_dataset import SM_Dataset
from model_lstm import LSTMMachine
from nn_cross_val import cross_val
from train import train
from test import test


def lstm():
    ds = SM_Dataset(data_type="vb")
    model = LSTMMachine(input_size=ds.X.shape[1])
    train(model, ds)
    r2 = test(model, ds)
    print(r2)


if __name__ == "__main__":
    lstm()