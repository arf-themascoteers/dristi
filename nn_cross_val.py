from sklearn.model_selection import KFold
from sm_dataset import SM_Dataset
from train import train
from test import test
import numpy as np


def cross_val(model_class, X, y, k):
    kf = KFold(n_splits=k)
    r2s = []
    mses = []
    maes = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        ds = SM_Dataset(X=X_train, y=y_train)
        model = model_class(ds.X.shape[1])
        model = train(model, ds)
        ds = SM_Dataset(X=X_test, y=y_test)
        r2, mse, mae = test(model, ds)
        print(f"{model_class.__name__} - Run#{i+1}: R2 = {r2}")
        r2s.append(r2)
        mses.append(mse)
        maes.append(mae)

    return np.array(r2s), np.array(mses), np.array(maes)


if __name__ == "__main__":
    pass