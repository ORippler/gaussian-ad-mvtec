"""Supervised classifier model (possibly pretrained)."""
import argparse
from collections import OrderedDict

from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

import pytorch_lightning as pl
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, RandomSampler

from ..common.augmentation import (
    dataset_aug,
    Compose,
    unnormalize,
    Normalize01,
)
from albumentations import Resize, Normalize
from ..common.dataset import StratifiedFoldSplit, AnomalyDetectionDataset
from ..common.evaluation import (
    log_roc_figure,
    latent_tsne_figure,
    latent_pca_figure,
    MinMaxSaver,
)
from ..common.utils import flatten
from .transparent import initialize_model, MODEL_NAMES
from typing import Tuple, Type, Iterable


class Classifier(pl.LightningModule):
    """Supervised classifier model (possibly pretrained)."""

    def __init__(
        self,
        hparams: argparse.Namespace,
        dataset_cls: Type[AnomalyDetectionDataset],
    ) -> None:
        super(Classifier, self).__init__()
        self.hparams = hparams
        self.model, self.input_size = initialize_model(
            model_name=hparams.arch,
            num_classes=1,
            use_pretrained=hparams.pretrained,
        )

        if hparams.pretrained:
            # ImageNet Normalization and Denormalization
            normalize, self.unnormalize = Normalize, unnormalize
        else:
            # Normalize between Zero and One
            normalize, self.unnormalize = Normalize01, lambda x: x
        final_transform = [normalize(), ToTensorV2()]
        final_eval_transform = copy.deepcopy(final_transform)

        if hparams.augment:
            uncached = Compose(
                [dataset_aug(hparams, dataset_cls), *final_transform]
            )
        else:
            uncached = Compose(final_transform)

        dataset = dataset_cls(
            hparams=self.hparams,
            train=True,
            supervised_train=True,
            transform=Compose([Resize(self.input_size, self.input_size)]),
            uncached_transform=Compose(uncached),
            load_masks=True,
        )
        self.datasplit = StratifiedFoldSplit(
            dataset,
            self.hparams.fold,
            test=True,
            uncached_eval_transform=Compose(final_eval_transform),
        )

        self.false_positives = []
        self.false_negatives = []
        self.test_saver = MinMaxSaver(unnormalize_fn=self.unnormalize)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        features, pred = self.model(x)

        return (features, pred)

    def training_step(self, batch: dict, batch_idx) -> dict:
        images, target = batch["image"], batch["target"]
        output = self(images)
        if isinstance(output, tuple):
            output = output[-1]
            # Transparent model, just take the final classifier output.
        loss_val = F.binary_cross_entropy_with_logits(
            output, target.view(target.size(0), -1).float()
        )
        tqdm_dict = {"loss/train": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        return output

    def validation_step(self, batch: dict, batch_idx) -> dict:
        images, target = batch["image"], batch["target"]
        output = self(images)
        result = {}
        if isinstance(output, tuple):
            features, output = output
            result["features"] = features
        pred = torch.sigmoid(output).squeeze(dim=1)

        if self.current_epoch % 10 == 9:
            for prediction, actual, image in zip(pred, target, images):
                if (
                    len(self.false_positives) < 4
                    and actual == 0
                    and prediction > 0.5
                ):
                    self.false_positives.append(self.unnormalize(image))
                elif (
                    len(self.false_negatives) < 4
                    and actual == 1
                    and prediction < 0.5
                ):
                    self.false_negatives.append(self.unnormalize(image))

        loss_val = F.binary_cross_entropy_with_logits(
            output, target.view(target.size(0), -1).float()
        )

        result.update(
            {"loss/val": loss_val, "prediction": pred, "target": target}
        )
        return result

    def validation_end(self, outputs: Iterable[dict]) -> dict:
        tqdm_dict = {}

        # Save mispredicted images.
        if self.false_positives:
            self.logger.experiment.add_image(
                "false_positives",
                make_grid(
                    torch.cat(
                        [fp.unsqueeze(0) for fp in self.false_positives]
                    ),
                    nrow=len(self.false_positives),
                ),
                self.current_epoch,
            )
            self.false_positives = []
        if self.false_negatives:
            self.logger.experiment.add_image(
                "false_negatives",
                make_grid(
                    torch.cat(
                        [fn.unsqueeze(0) for fn in self.false_negatives]
                    ),
                    nrow=len(self.false_negatives),
                ),
                self.current_epoch,
            )
            self.false_negatives = []

        # Compute & draw roc.
        pred = torch.cat([output["prediction"].cpu() for output in outputs], 0)
        target = torch.cat([output["target"].cpu() for output in outputs], 0)
        tqdm_dict["auc/val"] = log_roc_figure(
            self.logger, target, pred, self.current_epoch, data="val"
        )

        for metric_name in ["loss/val"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]
                metric_total += metric_value
            tqdm_dict[metric_name] = metric_total / len(outputs)

        # Feature space evaluation.
        if "features" in outputs[0] and (target != 0).any():
            features = torch.cat(
                [output["features"].cpu() for output in outputs], 0
            )
            tsne_fig = latent_tsne_figure(target, features)
            self.logger.experiment.add_figure(
                "latent/tsne/val", tsne_fig, self.current_epoch
            )
            pca_fig = latent_pca_figure(target, features)
            self.logger.experiment.add_figure(
                "latent/pca/val", pca_fig, self.current_epoch
            )
        tqdm_dict = flatten(tqdm_dict)

        return {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "loss/val": tqdm_dict["loss/val"],
            "latent_energy/val": tqdm_dict.get(
                "latent_energy/val/anomalies", torch.tensor(0)
            ),  # Tensor for PTL.
            "auroc/val": tqdm_dict["auc/val/ROC/anomalies"],
        }

    def test_step(self, batch: dict, batch_idx) -> dict:
        images, target = batch["image"], batch["target"]
        output = self(images)
        result = {}
        if isinstance(output, tuple):
            features, output = output
            # Transparent model, pass along the features.
            result["features"] = features

        pred = torch.sigmoid(output).squeeze(dim=1)

        self.test_saver.update(images, target, pred)

        result.update({"prediction": pred, "target": target})
        return result

    def test_end(self, outputs: Iterable[dict]) -> dict:
        self.logger.experiment.add_image(
            "min_max_good_pred_images/test",
            self.test_saver.good_grid(),
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            "min_max_anomaly_pred_images/test",
            self.test_saver.anomaly_grid(),
            self.current_epoch,
        )

        # Compute & draw roc.
        pred = torch.cat([output["prediction"].cpu() for output in outputs], 0)
        target = torch.cat([output["target"].cpu() for output in outputs], 0)
        tqdm_dict = {}
        tqdm_dict["auc/test"] = log_roc_figure(
            self.logger, target, pred, self.current_epoch
        )
        print("test auc: {}".format(tqdm_dict["auc/test"]))

        # Feature space evaluation.
        if "features" in outputs[0]:
            features = torch.cat(
                [output["features"].cpu() for output in outputs], 0
            )

            tsne_fig = latent_tsne_figure(target, features)
            self.logger.experiment.add_figure(
                "latent/tsne/test", tsne_fig, self.current_epoch
            )
            pca_fig = latent_pca_figure(target, features)
            self.logger.experiment.add_figure(
                "latent/pca/test", pca_fig, self.current_epoch
            )

        tqdm_dict = flatten(tqdm_dict)
        return {"progress_bar": tqdm_dict, "log": tqdm_dict}

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        trainset = self.datasplit.train()
        sampler = RandomSampler(trainset)
        print("Training with {} images".format(len(trainset)))
        return DataLoader(
            trainset,
            batch_size=self.hparams.batch_size,
            num_workers=4,
            sampler=sampler,
        )

    @pl.data_loader
    def val_dataloader(self) -> DataLoader:
        valset = self.datasplit.val()
        print("Validating with {} images".format(len(valset)))
        return DataLoader(
            valset, batch_size=self.hparams.batch_size, num_workers=2
        )

    @pl.data_loader
    def test_dataloader(self) -> DataLoader:
        testset = self.datasplit.test()
        print("Testing with {} images".format(len(testset)))
        return DataLoader(testset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """
        Specify the hyperparams for this LightningModule
        """
        parser.add_argument(
            "-a",
            "--arch",
            metavar="ARCH",
            default="resnet18",
            choices=MODEL_NAMES,
            help="model architecture: "
            + " | ".join(MODEL_NAMES)
            + " (default: resnet18)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="seed for initializing training. ",
        )
        parser.add_argument(
            "-b", "--batch_size", default=64, type=int, metavar="N"
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=0.001,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--augment",
            action="store_true",
            help="Enable data augmentation or not",
        )
        parser.add_argument(
            "--pretrained",
            action="store_true",
            help="Whether or not to use pre-trained model",
        )
        return parser
