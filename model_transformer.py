import torch
import torch.nn as nn


class TransformerMachine(nn.Module):
    def __init__(self, input_size):
        super(TransformerMachine, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=3)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=3)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=3)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x