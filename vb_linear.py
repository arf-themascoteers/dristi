from sklearn.linear_model import LinearRegression
from sm_dataset import SM_Dataset
from sklearn.model_selection import cross_val_score


def linear():
    ds = SM_Dataset(dstype="all")
    reg = LinearRegression()

    r2 = cross_val_score(reg, ds.X, ds.y, cv=10, scoring="r2")
    print("R-squared values:")
    print("\n".join(map(str, r2)))
    print("\n")
    print("Mean R-squared",r2.mean())

    mse = - cross_val_score(reg, ds.X, ds.y, cv=10, scoring="neg_mean_squared_error")
    print("MSE values:")
    print("\n".join(map(str, mse)))
    print("\n")
    print("Mean MSE",mse.mean())

    mae = - cross_val_score(reg, ds.X, ds.y, cv=10, scoring="neg_mean_absolute_error")
    print("MSE values:")
    print("\n".join(map(str, mae)))
    print("\n")
    print("Mean MAE",mae.mean())


if __name__ == "__main__":
    linear()