import torch.nn as nn

class UNetGenerator(nn.module):
    def __init__(self, in_channels, out_channels, n_downs, use_batch_norm=True, use_dropout=True):
        super(UNetGenerator, self).__init__()