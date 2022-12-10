import torch
from sklearn.linear_model import LinearRegression
from sm_dataset import SM_Dataset
from model_lstm import LSTMMachine
from nn_cross_val import cross_val
from train import train
from test import test


def lstm():
    for data_type in ["vb", "emd"]:
        ds = SM_Dataset(data_type=data_type)
        X = ds.X.detach().numpy()
        y = ds.y.detach().numpy()
        r2, mse, mae = cross_val(LSTMMachine, X, y, 10)

        print(data_type)
        print("-----------")

        print("R-squared values:\n\t", end="")
        print("\n\t".join(map(str, r2)))
        print("\n")
        print("Mean R-squared",r2.mean())
        print("\n")
        print("\n")

        print("MSE values:\n\t", end="")
        print("\n\t".join(map(str, mse)))
        print("\n")
        print("Mean MSE",mse.mean())
        print("\n")
        print("\n")

        print("MAE values:\n\t", end="")
        print("\n\t".join(map(str, mae)))
        print("\n")
        print("Mean MAE",mae.mean())
        print("\n")
        print("\n")

        print("End of:", data_type)

    print("All done. Bye.")


if __name__ == "__main__":
    lstm()