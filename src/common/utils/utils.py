"""Useful small utilities unrelated to the ML pipeline."""

import torch
import collections
import os
from typing import Union, Generator


def scantree(
    path: Union[str, bytes, os.PathLike]
) -> Generator[os.DirEntry, None, None]:
    """Recursively yield DirEntry objects for given directory."""
    for entry in sorted(os.scandir(path), key=lambda f: f.name):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry


def batched_index_select(
    input: torch.Tensor, index: torch.Tensor, dim: int
) -> torch.Tensor:
    """Index select a whole batch of inputs and indices along one dimension.

    See https://discuss.pytorch.org/t/batched-index-select/9115/7.

    Args:
        input (Tensor): Input of shape B x dim1 x dim2 x ...
        index (Tensor): Batch of indexes to select of shape B x dim_D.
        dim (int): Dimension D to select on.
    """
    views = [input.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def flatten(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = str(parent_key) + sep + k if parent_key else k
        if v and isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
