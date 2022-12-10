from sklearn.linear_model import LinearRegression
from sm_dataset import SM_Dataset
from model_cnn1d import CNN1DMachine
from nn_cross_val import cross_val
from train import train
from test import test


def rf():
    for data_type in ["vb", "emd"]:
        ds = SM_Dataset(data_type=data_type)

        model = CNN1DMachine()
        r2, mse, mae = cross_val(model, ds.X, ds.y, 10)

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


if __name__ == "__main__":
    ds = SM_Dataset(data_type="pca")
    model = CNN1DMachine(ds.X.shape[1])
    model = train(model, ds)
    r2, mse, mae = test(model, ds)
    print(r2, mse, mae)