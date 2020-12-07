from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.validation import indexable, _num_samples

# _num_samples only needed for fork of BaseCrossValidator in _three_way_split.

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
from typing import Optional, Callable, NoReturn, Generator
from collections.abc import Iterable
import argparse


class AnomalyDetectionDataset(VisionDataset, metaclass=ABCMeta):
    """A cached dataset class for anomaly detection experiments.

    This abstract base class should be inherited by all datasets supporting
    caching and transforms on the fly.

    Any instance of this must handle albumentations transforms.
    This means that the given transforms must be called as
    `transform(image=image)['image']`.

    Any subclass should implement the following function to retrieve the target
    quicker (without transforms on the image):

        def get_target(self, index, transform=False):
            \"""Faster version of __getitem__ for only the target.

            No transform on image needed.
            \"""

    """

    def __init__(
        self,
        root: str,
        hparams: Optional[argparse.Namespace] = None,
        train: bool = True,
        supervised_train: bool = False,
        load_masks: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache: bool = True,
        uncached_transform: Optional[Callable] = None,
    ) -> NoReturn:
        """
        Args:
            root (string): Directory with the extracted MVTec dataset images.
            hparams (argparse.Namespace): Params for this model run, including args
                from add_dataset_specific_args().
            train (bool): Load train data if True, test if False.
            supervised_train (bool): Classification dataset with all fully
                supervised defect labels during training (requires train).
            load_masks (bool): Whether to include ground truth anomaly
                segmentation maps (only if not train).
            transform (callable): transform to apply on dataset images
                (is cached between __getitem__ calls).
            target_transform (callable): transform to apply on dataset labels
                (is cached).
            cache (bool): whether they dataset fits into memory
            uncached_transform (callable): transform to apply on dataset images
                (not cached between __getitem__ calls).
        """
        super().__init__(
            root=root, transform=transform, target_transform=target_transform
        )
        self.hparams = hparams
        self.train = train
        self.supervised_train = supervised_train
        self.load_masks = load_masks
        self.target_transform = target_transform
        self.cache = cache
        self.uncached_transform = uncached_transform
        if self.load_masks and self.train and not self.supervised_train:
            raise ValueError(
                "Should not load masks in training, " "would only be zeros!"
            )

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Return the data item at index as a dict.

        Format is {"image": ..., "target": ..., ...}.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of this dataset (number of samples)."""
        pass

    @staticmethod
    def add_dataset_specific_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """Add the dataset-specific args to the given argument parser.

        Returns the parser.
        """
        return parser

    @staticmethod
    def augmentation_info(hparams: argparse.Namespace) -> dict:
        """Augmentation information for training with this dataset."""
        return {
            # Horizontal flips
            "flip": True,
            # 90° rotations (left or right)
            "rotate90": True,
            # up to 45° rotations (left or right)
            "rotate45": True,
            # Some patterns are really broken with shifts/scales on the edges.
            # background_edge means the edge is not important.
            "background_edge": True,
            # Apply noise
            "noise": True,
        }

    @staticmethod
    @abstractmethod
    def anomaly_info(hparams: argparse.Namespace) -> dict:
        """Meta-information on anomalies for training with this dataset.

        Returned dict should contain:
            "min_size": minimal size of an anomaly, as fraction of the total
                image area.
        """
        pass


class FoldSplit:
    """A k-fold cross validation splitter for a pytorch dataset."""

    # number of folds
    K = 5

    class TrainValSet(Dataset):
        """Internal train val set that comes out of the train/val split."""

        def __init__(
            self,
            dataset: AnomalyDetectionDataset,
            indices: Iterable,
            uncached_transform: Optional[Callable] = None,
        ):
            self.dataset = dataset
            self.indices = indices
            self.uncached_transform = uncached_transform

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int) -> dict:
            if self.uncached_transform is not None:
                backup = self.dataset.uncached_transform
                self.dataset.uncached_transform = self.uncached_transform
            item = self.dataset[self.indices[index]]
            if self.uncached_transform is not None:
                self.dataset.uncached_transform = backup
            return item

    def __init__(
        self,
        dataset: AnomalyDetectionDataset,
        fold: int,
        test: bool = False,
        uncached_eval_transform: Optional[Callable] = None,
    ) -> NoReturn:
        """Init a CrossValSplit on the given dataset.

        Args:
            fold (int): Index of the current split (the i-th out of K folds).
            test (bool): Include the a test set as in a (K-2/1/1) split, else
                (K-1/1).
            uncached_eval_transform (callable): transformation to replace the
                uncached_transform in dataset with for val/test.
        """
        self.dataset = dataset
        self.uncached_eval_transform = uncached_eval_transform
        assert 0 <= fold < self.K

        splits = self._get_splits(test=test)
        # self.set_indices = splits[fold] but it is a generator
        self.set_indices = next(s for i, s in enumerate(splits) if i == fold)

    def _get_splits(self, test: bool = False) -> Generator:
        """Get the split indices returned by sklearn."""
        splitter = KFold(n_splits=self.K, shuffle=True, random_state=0)
        if test:
            return self._three_way_split(splitter, np.zeros(len(self.dataset)))
        else:
            return splitter.split(np.zeros(len(self.dataset)))

    @staticmethod
    def _three_way_split(
        splitter: KFold, X, y: Optional = None, groups: Optional = None
    ) -> Generator:
        """A modified version of BaseCrossValidator.split().

        Yields (K-2/1/1) train/val/test splits.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        test_masks_it = splitter._iter_test_masks(X, y, groups)
        first_mask = last_mask = next(test_masks_it)
        for test_mask in test_masks_it:
            train_index = indices[
                np.logical_not(np.logical_or(test_mask, last_mask))
            ]
            val_index = indices[last_mask]
            test_index = indices[test_mask]
            yield train_index, val_index, test_index
            last_mask = test_mask
        # last fold
        test_mask = first_mask
        train_index = indices[
            np.logical_not(np.logical_or(test_mask, last_mask))
        ]
        val_index = indices[last_mask]
        test_index = indices[test_mask]
        yield train_index, val_index, test_index

    def train(self) -> TrainValSet:
        """Return a pytorch dataset for the training set."""
        return self.TrainValSet(self.dataset, self.set_indices[0])

    def val(self) -> TrainValSet:
        """Return a pytorch dataset for the validation set."""
        return self.TrainValSet(
            self.dataset,
            self.set_indices[1],
            uncached_transform=self.uncached_eval_transform,
        )

    def test(self) -> TrainValSet:
        """Return a pytorch dataset for the test set."""
        if len(self.set_indices) != 3:
            raise ValueError("Cannot use test() with test=False")
        return self.TrainValSet(
            self.dataset,
            self.set_indices[2],
            uncached_transform=self.uncached_eval_transform,
        )


class StratifiedFoldSplit(FoldSplit):
    """A stratified k-fold cross validation splitter for a pytorch dataset."""

    def _get_splits(self, test: bool = False) -> Generator:
        """Get the split indices returned by sklearn."""
        splitter = StratifiedKFold(
            n_splits=self.K, shuffle=True, random_state=0
        )
        # aggregate targets for splitting
        get_target = getattr(self.dataset, "get_target", None)
        if callable(get_target):
            targets = [get_target(i) for i in range(len(self.dataset))]
        else:
            targets = [s[1] for s in self.dataset]
        print(
            "Splitting on {} anomalies".format(
                sum(1 for t in targets if t != 0)
            )
        )
        if test:
            return self._three_way_split(
                splitter, np.zeros(len(self.dataset)), targets
            )
        else:
            return splitter.split(np.zeros(len(self.dataset)), targets)
