import torch
from torch.utils.data import DataLoader
from sm_dataset import SM_Dataset
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def test(model, ds):
    BATCH_SIZE = 2000
    dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    #print(f"Test started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            return r2, mse, mae

if __name__ == "__main__":
    test()