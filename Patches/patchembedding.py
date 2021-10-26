import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbeddings(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_size : int = 768,
                 image_size : int = 224):

        super().__init__()

        self._patch_size = patch_size
        self._in_channels = in_channels
        self._embedding_size = embedding_size
        self._image_size = image_size

        self._cls_token = nn.Parameter(torch.randn(1, 1, self._embedding_size))
        self._positions = nn.Parameter(torch.randn((
            self._image_size//self._patch_size)**2 + 1, self._embedding_size
        ))


        self._projections = nn.Sequential(
            nn.Conv2d(self._in_channels,
                      self._embedding_size,
                      kernel_size= (self._patch_size, self._patch_size),
                      stride= (self._patch_size, self._patch_size)),
            Rearrange('b e (h) (w) -> b (h w) e')
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor :
        #print(x.shape)
        b, _, _, _ = x.shape
        x = self._projections(x)
        class_tokens = repeat(self._cls_token, '() n e -> b n e', b=b)
        x = torch.cat([class_tokens, x], dim = 1)
        x += self._positions
        return x