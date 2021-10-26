import torch.nn as nn
from Attention.attention import MultiHeadAttention
from Patches.patchembedding import PatchEmbeddings
from Models.blocks import ClassificationBlock


class ViT(nn.Sequential):
    def __init__(self,
                 in_channels : int = 3,
                 patch_size : int = 16,
                 embedding_size : int = 64,
                 image_size : int = 224,
                 depth : int = 16,
                 n_classes : int = 2,
                 **kwargs):

        super().__init__(
            PatchEmbeddings(in_channels=in_channels, patch_size=patch_size, embedding_size=embedding_size, image_size=image_size),
            MultiHeadAttention(embedding_size, depth, **kwargs),
            ClassificationBlock(embedding_size, n_classes)
        )
