import argparse
from collections import namedtuple
import itertools
import os
import random

import cv2

import numpy as np

from ..common.augmentation import dataset_aug
from ..datasets.mvtecad import MVTecAD


def grid_image(imgs, margin=None):
    """Align images in a grid and return a single big image.

    imgs must be a list of lists of images.
    See https://gist.github.com/pgorczak/95230f53d3f140e4939c.
    """
    # if any(i.shape != imgs[0][0].shape for j in range(imgs) for i in imgs[j][1:]):
    #    raise ValueError('Not all images have the same shape.')

    h, w = len(imgs), len(imgs[0])
    img_h, img_w, img_c = imgs[0][0].shape

    m_x = 0
    m_y = 0
    if margin is not None:
        if "." in margin:
            m = float(margin)
            m_x = int(m * img_w)
            m_y = int(m * img_h)
        else:
            m_x = int(margin)
            m_y = m_x

    imgmatrix = np.zeros(
        (img_h * h + m_y * (h - 1), img_w * w + m_x * (w - 1), img_c), np.uint8
    )

    imgmatrix.fill(255)

    positions = itertools.product(range(w), range(h))
    for x_i, y_i in positions:
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y : y + img_h, x : x + img_w, :] = imgs[y_i][x_i]
    return imgmatrix


def category_grid(
    categories,
    augment=False,
    anomalies=False,
    segment_border=False,
    columns=8,
    width=512,
):
    """Return a category grid image with one category per row and width images per column."""
    grid = []
    for category in categories:
        hparams = namedtuple("hparams", ["category"])
        dataset = MVTecAD(
            hparams=hparams(category),
            train=not anomalies,
            cache=False,
            load_masks=segment_border,
        )
        indices = range(len(dataset))
        if anomalies:
            # Select only anomalies.
            indices = filter(
                lambda i: dataset.get_target(i) != "good", indices
            )
        # Randomly select columns images.
        indexlist = random.sample(list(indices), k=columns)
        images = map(lambda i: dataset[i], indexlist)
        if segment_border:
            if not anomalies:
                raise ValueError(
                    "Cannot draw segmentation border without anomalies"
                )

            # Draw a border around the ground truth segmentation.
            def draw_border(item):
                image, mask = item["image"], item["mask"]
                kernel = np.ones((5, 5), np.uint8)
                # Dilate & erode the ground truth by 2 pixels
                gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
                image[gradient != 0] = (255, 0, 0)
                return image

            images = map(draw_border, images)
        else:
            images = map(lambda item: item["image"], images)
        if augment:
            aug = dataset_aug(hparams(category), MVTecAD, p=1)
            images = map(lambda image: aug(image=image)["image"], images)
        if width is not None:
            # Downscale to resolution.
            images = map(
                lambda image: cv2.resize(
                    image, (width, width), interpolation=cv2.INTER_AREA
                ),
                images,
            )

        grid.append(list(images))
    return grid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--outfile", type=str, default="./grid.jpg")
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--anomalies", action="store_true")
    parser.add_argument("--segment_border", action="store_true")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    if args.categories is None:
        args.categories = MVTecAD.available_categories
    grid = grid_image(
        category_grid(
            args.categories,
            augment=args.augment,
            columns=args.columns,
            anomalies=args.anomalies,
            segment_border=args.segment_border,
        )
    )
    cv2.imwrite(args.outfile, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
