import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 embedding_size : int = 768,
                 num_heads : int = 8,
                 dropout : float = 0.0):
        super().__init__()

        self._embedding_size = embedding_size
        self._num_heads = num_heads
        self._dropout = dropout

        self._query_key_value = nn.Linear(self._embedding_size, self._embedding_size * 3)
        self._attention_dropout = nn.Dropout(self._dropout)
        self._projection = nn.Linear(self._embedding_size, self._embedding_size)

    def forward(self, x: torch.Tensor, mask : torch.Tensor = None) -> torch.Tensor:

        qkv = rearrange(self._query_key_value(x), 'b n (h d qkv) -> (qkv) b h n d', h = self._num_heads, qkv=3)
        queries , keys, values = qkv[0], qkv[1], qkv[2]

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill)

        scaling = self._embedding_size ** 0.5
        attention = F.softmax(energy, dim=-1) / scaling
        attention = self._attention_dropout(attention)

        output = torch.einsum('bhal, bhlv -> bhav', attention, values)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self._projection(output)

        return output
