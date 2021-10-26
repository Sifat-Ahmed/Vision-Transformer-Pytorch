import torch.nn as nn
from einops.layers.torch import Reduce
from einops import reduce

class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 embedding_size : int = 768,
                 expansion : int = 4,
                 drop_proba : float = 0.0):


        self._embedding_size = embedding_size
        self._expansion = expansion
        self._drop_proba = drop_proba

        super().__init__(
            nn.Linear(self._embedding_size, self._expansion * self._embedding_size),
            nn.GELU(),
            nn.Dropout(self._drop_proba),
            nn.Linear(self._expansion * self._embedding_size, self._expansion)
        )


class ClassificationBlock(nn.Sequential):
    def __init__(self,
                 embedding_size : int = 768,
                 n_classes: int = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, n_classes)
        )