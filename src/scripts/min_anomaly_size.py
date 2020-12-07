#!/usr/bin/env python3
"""Calculate the minimum anomaly size per category."""

import argparse
from collections import namedtuple

import numpy as np

from skimage.measure import label

from ..datasets.mvtecad import MVTecAD


def component_sizes(mask):
    """Compute the component sizes in the given masks using connected component analysis.

    Sizes are given as fraction of image area.
    """
    labels = label(mask, background=0)
    unique, counts = np.unique(labels, return_counts=True)
    # Exclude 0 (background) for the returned component sizes
    # and scale & save as proportion of image size.
    return counts[unique != 0] / (mask.shape[0] * mask.shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    args = parser.parse_args()

    if args.categories is None:
        args.categories = MVTecAD.available_categories

    Params = namedtuple("hparams", ["category"])

    # Generate a nice python dict for each category.
    print("min_anomaly_sizes = {")
    for category in args.categories:
        dataset = MVTecAD(
            hparams=Params(category), train=False, cache=False, load_masks=True
        )
        # only load masks
        masks = (dataset.get_mask(i) for i in range(len(dataset)))
        min_size = min(
            component_sizes(mask).max() for mask in masks if mask is not None
        )
        print('    "{}": {},'.format(category, min_size))
    print("}")
