import torch
import torch.nn as nn


class CNN1DMachine(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        FC_INPUT = 0
        if input_size == 63:
            FC_INPUT = 64
        elif input_size == 126:
            FC_INPUT = 176

        self.band_net = nn.Sequential(
            nn.Conv1d(1,8,6),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(8, 16, 6),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(FC_INPUT,1)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.band_net(x)
        return x