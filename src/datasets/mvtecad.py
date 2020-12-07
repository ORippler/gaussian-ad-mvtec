import os

import cv2

import numpy as np

from ..common.dataset import AnomalyDetectionDataset
from typing import Optional, Callable, Tuple, NoReturn, Union
import argparse


class MVTecAD(AnomalyDetectionDataset):
    """MVTec Anomaly Detection dataset.

    Uses the albumentations transformation pipeline (instead of torchvision).
    So make sure to NOT use torchvision.transforms.

    By default has a completely deterministic (sorted) list of images,
    so you must use shuffle=True in the dataloader to not have all anomalies at the end.
    """

    available_categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    objects = [
        "bottle",
        "cable",
        "capsule",
        "hazelnut",
        "metal_nut",
        "pill",
        "screw",
        "toothbrush",
        "transistor",
        "zipper",
    ]
    """Object categories."""

    textures = ["carpet", "grid", "leather", "tile", "wood"]
    """Texture categories."""

    flip_aligned_hor = ["cable", "capsule", "metal_nut", "pill", "screw"]
    """Aligned to horizontal flips, i.e. all images are correctly flipped."""

    flip_symmetric_hor = [
        "bottle",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]
    """Symmetric to horizontal flips, i.e. indistinguishable if flipped."""

    rotation_aligned = [
        "cable",
        "capsule",
        "pill",
        "toothbrush",
        "transistor",
        "zipper",
    ]
    """Objects that are aligned to point upwards and may not be rotated at all."""

    rotation_aligned_90 = [
        "cable",
        "capsule",
        "pill",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]
    """Objects that are aligned to 90 degree rotations in either direction."""

    rotation_aligned_180 = [
        "cable",
        "capsule",
        "pill",
        "toothbrush",
        "transistor",
        "zipper",
    ]
    """Objects that are aligned to 180 degree rotations, i.e. they have an 'up' side."""

    rotation_symmetric = ["bottle", "leather", "metal_nut", "tile"]
    """Completely rotation symmetric, i.e. a rotation by any angle is indistinguishable."""

    rotation_symmetric_90 = [
        "bottle",
        "carpet",
        "leather",
        "metal_nut",
        "tile",
    ]
    """Rotation symmetric categories (in the normal case), i.e. a rotation by 90 degrees in either direction is indistinguishable."""

    rotation_symmetric_180 = [
        "bottle",
        "carpet",
        "leather",
        "metal_nut",
        "tile",
        "wood",
    ]
    """Rotation symmetric categories (in the normal case), i.e. a rotation by 180 degrees is indistinguishable."""

    def __init__(
        self,
        root: str = os.getenv("MVTECPATH", "/images/PublicDataset/Mvtec_AD"),
        hparams: Optional[argparse.Namespace] = None,
        train: bool = True,
        supervised_train: bool = False,
        load_masks: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Callable = lambda class_name: 0
        if class_name == "good"
        else 1,
        cache: bool = True,
        uncached_transform: Optional[Callable] = None,
    ):
        """
        Args:
            root (string): Directory with the extracted MVTec dataset images.
            hparams (argparse.Namespace): Params for this model run, including args from add_dataset_specific_args().
            train (bool): Load train data if True, test if False.
            supervised_train (bool): Classification dataset with all fully supervised defect labels during training (requires train).
            load_masks (bool): Whether to include ground truth anomaly segmentation maps (only if not train).
            transform (callable): transform to apply on dataset images (is cached between __getitem__ calls).
            target_transform (callable): transform to apply on dataset labels (is cached).
            uncached_transform (callable): transform to apply on dataset images (not cached between __getitem__ calls).
        """
        super().__init__(
            root=root,
            train=train,
            supervised_train=supervised_train,
            load_masks=load_masks,
            transform=transform,
            target_transform=target_transform,
            cache=cache,
            uncached_transform=uncached_transform,
        )

        categories = [
            hparams.category
        ]  # The following allows multiple categories.
        if categories is None:
            self.categories = self.available_categories
        else:
            if not (set(categories) <= set(self.available_categories)):
                raise ValueError(
                    "Given categories not a subset of {}".format(
                        self.available_categories
                    )
                )
            self.categories = sorted(
                categories
            )  # Don't be needlessly undeterministic on swapped categories.

        # Load data from root
        self.classes = set()
        self.samples = (
            []
        )  # List(Tuple(path, class)) or List(Tuple(image, class)) or List(Tuple(image, class, mask))

        def image_to_mask(path: str) -> str:
            """Return the corresponding mask path for the given image path."""
            root, suffix = path.rsplit(
                "test", 1
            )  # replace last 'test' folder with 'ground_truth'
            path, ext = suffix.rsplit(".", 1)
            return root + "ground_truth" + path + "_mask." + ext

        def load_to_cache_with_mask(
            t: Tuple[str, str, str]
        ) -> Tuple[np.ndarray, str, np.ndarray]:
            """Load tuple t to cache including mask."""
            image, mask = self.load_transform(t[0], mask_path=t[2])
            return image, t[1], mask

        def add_class_images(class_folder: os.DirEntry) -> NoReturn:
            """Add a fraction of the images and masks in class_folder to the samples."""
            images = filter(
                lambda f: f.is_file() and f.name.endswith(".png"),
                os.scandir(class_folder.path),
            )
            imagelist = sorted(images, key=lambda e: e.name)

            images = map(lambda f: (f.path, class_folder.name), imagelist)

            # Load matching masks for the test set (only in anomaly categories).
            if load_masks:
                if class_folder.name == "good":
                    images = map(lambda t: (t[0], t[1], None), images)
                else:
                    images = map(
                        lambda t: (t[0], t[1], image_to_mask(t[0])), images
                    )
                # Cache images (and move to GPU).
                if cache:
                    images = map(load_to_cache_with_mask, images)
            elif cache:
                images = map(
                    lambda t: (self.load_transform(t[0])[0], t[1]), images
                )

            self.samples.extend(images)

        train_test = ["train" if train else "test"]
        if supervised_train:
            train_test.append("test")
            if not train:
                raise ValueError("Cannot have supervised_train and not train")

        for category in self.categories:
            for dataset in train_test:
                category_path = os.path.join(root, category, dataset)
                # classes must be sorted to have completely deterministic dataset generation
                classes = sorted(
                    filter(lambda f: f.is_dir(), os.scandir(category_path)),
                    key=lambda e: e.name,
                )
                self.classes.update(map(lambda f: f.name, classes))
                for class_folder in classes:
                    add_class_images(class_folder)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        """
        Args:
            index (int): Index
        Returns:
            dict("sample", "target", "mask") where target is class_name of the target class
                (i.e. 'good' or some defect).
        """
        sample, target = self.samples[index][0], self.samples[index][1]
        mask = None
        if len(self.samples[index]) > 2:
            mask = self.samples[index][2]
        if not self.cache:
            sample, mask = self.load_transform(sample, mask_path=mask)
        if self.target_transform is not None:
            target = self.target_transform(target)

        item = {"image": sample, "target": target}
        if mask is not None:
            item["mask"] = mask
        # Transform mask and image at the same time (same random transform).
        if self.uncached_transform is not None:
            item = self.uncached_transform(**item)
        return item

    def get_target(
        self, index: int, transform: bool = True
    ) -> Union[str, int]:
        """Faster version of __getitem__ for only the target (no transform on sample needed)."""
        target = self.samples[index][1]
        if self.target_transform is not None and transform:
            target = self.target_transform(target)
        return target

    def get_mask(self, index: int) -> np.ndarray:
        """Faster version of __getitem__ for only the mask (no transform on image needed).

        Returns None if no mask exists (contrary to getitem, which returns a
        zero mask).
        """
        if len(self.samples[index]) < 3:
            raise ValueError("No masks loaded. Use load_masks=True")
        mask = self.samples[index][2]
        if not self.cache:
            if mask is not None:
                mask = self.load_mask(mask)
            if self.transform is not None:
                mask = self.transform(mask=mask)["mask"]
        if self.uncached_transform is not None:
            mask = self.uncached_transform(mask=mask)["mask"]
        return mask

    def load_transform(
        self, path: str, mask_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read the image from the given path and transform it with self.transform.

        If mask_path is given, load the mask as well.
        If self.load_masks it will include zero masks where no mask_path is given.

        Returns a tuple(image, mask) or (image, None).
        """
        image = cv2.imread(path)
        if image is None or image.size == 0:
            raise OSError("Could not read image: {}".format(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        item = {"image": image}
        if mask_path is not None:
            item["mask"] = self.load_mask(mask_path)
        elif self.load_masks:  # load an empty mask and also transform it.
            item["mask"] = np.zeros(
                (image.shape[0], image.shape[1]), dtype=np.uint8
            )
        if self.transform is not None:
            item = self.transform(**item)
        return item["image"], item.get("mask", None)

    def load_mask(self, path: str) -> np.ndarray:
        """Load the mask from the given path, or raise an error."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size == 0:
            raise OSError("Could not read mask: {}".format(path))
        return mask

    @staticmethod
    def add_dataset_specific_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add the dataset-specific args to the given argument parser and return it."""
        parser.add_argument(
            "--category",
            type=str,
            choices=MVTecAD.available_categories,
            required=True,
            help="The MVTec AD category to train and test on",
        )
        return parser

    @staticmethod
    def augmentation_info(hparams: argparse.Namespace) -> dict:
        """Augmentation used for anomaly detection tasks on an MVTec category."""
        return {
            # Some MVTec categories contain text or directions so they must not be flipped
            "flip": (
                hparams.category
                not in ["metal_nut", "pill", "capsule", "screw"]
            ),
            # Some categories may not be rotated (because a 90° rotation is an anomaly)
            "rotate90": (hparams.category not in ["transistor"]),
            # Some smaller rotations produce edge artifacts for some categories
            "rotate45": (
                hparams.category
                not in [
                    "bottle",
                    "cable",
                    "capsule",
                    "metal_nut",
                    "pill",
                    "screw",
                    "toothbrush",
                    "transistor",
                    "wood",
                    "zipper",
                ]
            ),
            # Some patterns are really messed up with shifts/scales on the edges
            "background_edge": (hparams.category not in ["grid"]),
            "noise": True,
        }

    @staticmethod
    def anomaly_info(hparams: argparse.Namespace) -> dict:
        """Meta-information on anomalies for training with this dataset.

        Returned dict should contain:
            "min_size": minimal size of an anomaly, as fraction of the image
                area.
            "size_quantiles": q = 5 quantiles of anomaly sizes, given as q + 1
                boundary values in increasing order.
        """
        # Generated from src.scripts.min_anomaly_size.
        min_anomaly_sizes = {
            "bottle": 0.00575679012345679,
            "cable": 0.0014362335205078125,
            "capsule": 0.000371,
            "carpet": 0.001499176025390625,
            "grid": 0.0007152557373046875,
            "hazelnut": 0.0023288726806640625,
            "leather": 0.000858306884765625,
            "metal_nut": 0.0011959183673469387,
            "pill": 0.0003171875,
            "screw": 0.000659942626953125,
            "tile": 0.008421201814058957,
            "toothbrush": 0.00141143798828125,
            "transistor": 0.0022373199462890625,
            "wood": 0.001071929931640625,
            "zipper": 0.0015916824340820312,
        }
        # Generated from code.scripts.anomaly_size_distribution.
        return {"min_size": min_anomaly_sizes[hparams.category]}


# The DATASET variable controls which class is instantiated
DATASET = MVTecAD
