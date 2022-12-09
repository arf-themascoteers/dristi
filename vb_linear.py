from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sm_dataset import SM_Dataset
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def linear():
    ds = SM_Dataset(dstype="all")
    reg = LinearRegression()
    print(metrics.get_scorer_names())

    r2 = cross_val_score(reg, ds.X, ds.y, cv=10, scoring="r2")
    print(r2)
    print(r2.mean())

    mse = - cross_val_score(reg, ds.X, ds.y, cv=10, scoring="neg_mean_squared_error")
    print(mse)
    print(mse.mean())

    exit()
    return r2, mae, mse


if __name__ == "__main__":
    r2, mae, mse = linear()
    print(r2, mae, mse)