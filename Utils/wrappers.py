import torch.nn as nn

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, x, **kwargs):
        res = x

        x = self._fn(x, **kwargs)
        x += res

        return x
