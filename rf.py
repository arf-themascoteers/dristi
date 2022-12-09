from sklearn.linear_model import LinearRegression
from sm_dataset import SM_Dataset
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


def rf():
    for data_type in ["vb", "emd", "pca", "pca_emd"]:
        ds = SM_Dataset(data_type=data_type)

        reg = RandomForestRegressor()

        r2 = cross_val_score(reg, ds.X, ds.y, cv=10, scoring="r2")

        print(data_type)
        print("-----------")

        print("R-squared values:\n\t", end="")
        print("\n\t".join(map(str, r2)))
        print("\n")
        print("Mean R-squared",r2.mean())

        mse = - cross_val_score(reg, ds.X, ds.y, cv=10, scoring="neg_mean_squared_error")
        print("MSE values:\n\t", end="")
        print("\n\t".join(map(str, mse)))
        print("\n")
        print("Mean MSE",mse.mean())

        mae = - cross_val_score(reg, ds.X, ds.y, cv=10, scoring="neg_mean_absolute_error")
        print("MAE values:\n\t", end="")
        print("\n\t".join(map(str, mae)))
        print("\n")
        print("Mean MAE",mae.mean())


if __name__ == "__main__":
    rf()