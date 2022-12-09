from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sm_dataset import SM_Dataset


def linear():
    ds = SM_Dataset(is_train=True)
    reg = LinearRegression().fit(ds.X, ds.y)
    ds = SM_Dataset(is_train=False)
    y_hat = reg.predict(ds.X)
    r2 = r2_score(ds.y, y_hat)
    mae = mean_absolute_error(ds.y, y_hat)
    mse = mean_squared_error(ds.y, y_hat)

    return r2, mae, mse


if __name__ == "__main__":
    r2, mae, mse = linear()
    print(r2, mae, mse)