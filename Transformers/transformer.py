import torch.nn as nn
from Utils.wrappers import ResidualAdd
from Models.blocks import FeedForwardBlock
from Attention.attention import MultiHeadAttention

class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 emb_size : int = 768,
                 drop_proba : float = 0.0,
                 forward_expansion: int = 4,
                 forward_drop_proba : float = 0.0,
                 **kwargs):
        super().__init__()

        self._embedding_size = emb_size
        self._dropout_proba = drop_proba
        self._forward_expansion = forward_expansion
        self._forward_drop_proba = forward_drop_proba



        ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(self._embedding_size),
                MultiHeadAttention(self._embedding_size, **kwargs),
                nn.Dropout(self._dropout_proba)

            )
        )

        ResidualAdd(
            nn.Sequential(
                nn.LayerNorm(self._embedding_size),
                FeedForwardBlock( self._embedding_size, expansion = self._forward_expansion, drop_proba = self._forward_drop_proba),
                nn.Dropout(self._forward_drop_proba)
            )
        )
