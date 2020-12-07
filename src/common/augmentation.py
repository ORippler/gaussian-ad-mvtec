"""Augmentations for all models.

Uses the albumentations library and OpenCV to process augmentations.
This ensures fast enough speeds so that augmentations can be run online.
"""

from albumentations import (
    ImageOnlyTransform,
    BasicTransform,
    Compose,
    OneOf,
    MotionBlur,
    MedianBlur,
    Blur,
    CLAHE,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    HorizontalFlip,
    RandomRotate90,
    ShiftScaleRotate,
)

import numpy as np
import cv2

import torch
from typing import NoReturn, Type
import argparse
from .dataset import AnomalyDetectionDataset


class Normalize01(ImageOnlyTransform):
    """Divide pixel values by 255 = 2**8 - 1.

    Args:
        max_pixel_value (float): maximum possible pixel value

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        max_pixel_value: float = 255.0,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> NoReturn:
        super(Normalize01, self).__init__(always_apply, p)
        self.max_pixel_value = max_pixel_value

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        image = image.astype(np.float32)
        image /= self.max_pixel_value
        return image

    def get_transform_init_args_names(self) -> str:
        return "max_pixel_value"


class MaskOnlyTransform(BasicTransform):
    """Transform applied to mask only."""

    @property
    def targets(self) -> dict:
        return {"mask": self.apply_to_mask}

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(
            img,
            **{
                k: cv2.INTER_NEAREST if k == "interpolation" else v
                for k, v in params.items()
            }
        )


class MaskToSize(MaskOnlyTransform):
    """Transform mask to its fraction of the image area."""

    def __init__(self, always_apply: bool = False, p: float = 1.0) -> NoReturn:
        super().__init__(always_apply, p)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        result = (image != 0).sum() / (image.shape[0] * image.shape[1])
        # Need to create a 0D array from the mask size for ToTensor to work.
        return np.array(result)


def pixel_aug(p: float = 0.5, noise: bool = True) -> Compose:
    """Augmentation only on a pixel-level."""
    augs = [
        OneOf(
            [
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ],
            p=0.2,
        ),
        OneOf(
            [
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.2)),
            ],
            p=0.3,
        ),
        # Reduced hue shift to not change the color that much (purple
        # hazelnuts).
        # reduced val shift to not overly darken the image
        HueSaturationValue(
            hue_shift_limit=10, val_shift_limit=(-10, 20), p=0.3
        ),
    ]
    if noise:
        augs.append(
            OneOf(
                [
                    # Slightly less aggressive:
                    IAAAdditiveGaussianNoise(
                        scale=(0.01 * 255, 0.03 * 255), per_channel=False
                    ),
                    IAAAdditiveGaussianNoise(
                        scale=(0.01 * 255, 0.03 * 255), per_channel=True
                    ),
                ],
                p=0.2,
            )
        )
    return Compose(augs, p=p)


def detection_aug(
    p: float = 0.5,
    flip: bool = True,
    rotate90: bool = True,
    noise: bool = True,
    rotate45: bool = True,
    background_edge: bool = True,
) -> Compose:
    """Augmentation used for anomaly detection tasks."""
    augs = [
        pixel_aug(p=1, noise=noise),
    ]
    if flip:
        # Only horizontal flips as vertical is same as horizontal + 180°
        # rotate.
        augs.append(HorizontalFlip())
    if rotate90:
        augs.append(RandomRotate90())
    augs.append(
        ShiftScaleRotate(
            shift_limit=0.05 if background_edge else 0,
            scale_limit=(-0.05, 0.1 if background_edge else 0),
            rotate_limit=(45 if rotate45 else 15) if background_edge else 0,
            p=0.2,
        )
    )
    return Compose(augs, p=p)


def dataset_aug(
    hparams: argparse.Namespace,
    dataset_cls: Type[AnomalyDetectionDataset],
    p: float = 0.5,
    **kwargs
):
    """Build a detection_aug augmentation for the given dataset.

    kwargs overrides any augmentation feature defined in
    dataset_cls.augmentation_info().
    """
    info = dataset_cls.augmentation_info(hparams)
    info.update(kwargs)
    return detection_aug(p=p, **info)


def unnormalize(
    tensor: torch.Tensor,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225],
    inplace: bool = False,
) -> torch.Tensor:
    """Undo the operation of albumentations' Normalize.

    For showing (augmented) images.
    """
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor
