# Vision Transformer Implementation from Scratch
#### This repository demonstrates image classification with Vision Transformers. 
> Refer to ```config.py``` for model related parameters.

```
Folder Structure

>>>> Attention
    - __init__.py
    - attention.py
        + MultiHeadAttention
>>>> Dataset
    - __init__.py
    - dataset.py
        + SmokeDataset
    - utils.py
        + get_train_test()
>>>> Models
    - __init__.py
    - blocks.py
        + FeedForwardBlock
        + ClassificationBlock
>>>> Patches
    - __init__.py
    - patchembedding.py
        + PatchEmbeddings
>>>> Transformers
    - __init__.py
    - transformer.py
        + TransformerEncoder
>>>> Utils
    - __init__.py
    - helper.py
        + TqdmUpTo
        + MetricMonitor
        + calculate_accuracy
    - visualize.py
        + display_image_grid
        + visualize_augmentations
        + plot_curves
    - wrappers.py
        + ResidualAdd
>>>> config.py
>>>> main.py
```

