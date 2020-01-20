import torch
from torch import nn

MFCC_SIZE = 64
NUM_HIDDEN = 4
DROPOUT = 0.5


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=MFCC_SIZE, hidden_size=MFCC_SIZE, num_layers=NUM_HIDDEN, dropout=DROPOUT, bidirectional=True)

    def forward(self, mfcc):
        _, hidden = self.gru.forward(mfcc)
        return torch.cat((hidden[NUM_HIDDEN - 1], hidden[NUM_HIDDEN << 1 - 1]), dim=1)
