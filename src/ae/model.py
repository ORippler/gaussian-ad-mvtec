"""Standard Autoencoder model for anomaly detection with different losses."""

import argparse
from collections import OrderedDict
import copy

from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler
from ..common.augmentation import (
    dataset_aug,
    Compose,
    Normalize01,
)
from albumentations import Resize
from ..common.autoencoder import build_autoencoder, MODEL_NAMES
from ..common.dataset import (
    FoldSplit,
    StratifiedFoldSplit,
    AnomalyDetectionDataset,
)
from ..common.evaluation import (
    log_roc_figure,
    latent_tsne_figure,
    latent_pca_figure,
    MinMaxSaver,
    connected_component_rocs,
)
from ..common.utils import flatten
from torchvision.utils import make_grid
from typing import NoReturn, Tuple, Type, Iterable


class AE(pl.LightningModule):
    """Autoencoder model for anomaly detection with different losses."""

    def __init__(
        self,
        hparams: argparse.Namespace,
        dataset_cls: Type[AnomalyDetectionDataset],
    ) -> NoReturn:
        super(AE, self).__init__()
        self.hparams = hparams
        self.encoder, self.decoder, self.input_size = build_autoencoder(
            arch=hparams.arch, pretrained=hparams.pretrained
        )

        if hparams.loss_criterion == "l1":
            self.criterion = nn.L1Loss(reduction="none")
        elif hparams.loss_criterion == "mse":
            self.criterion = nn.MSELoss(reduction="none")
        else:
            raise ValueError(
                "Unavailable loss: {}".format(hparams.loss_criterion)
            )

        final_transform = [Normalize01(), ToTensorV2()]
        final_eval_transform = copy.deepcopy(final_transform)

        if hparams.augment:
            uncached = Compose(
                [
                    dataset_aug(hparams, dataset_cls, noise=False),
                    *final_transform,
                ]
            )
        else:
            uncached = Compose(final_transform)

        dataset = dataset_cls(
            hparams=hparams,
            train=True,
            transform=Resize(self.input_size, self.input_size),
            uncached_transform=uncached,
        )
        self.datasplit = FoldSplit(
            dataset,
            self.hparams.fold,
            uncached_eval_transform=Compose(final_eval_transform),
        )
        self.testset = dataset_cls(
            hparams=hparams,
            train=False,
            load_masks=True,
            transform=Resize(self.input_size, self.input_size),
            uncached_transform=Compose(final_eval_transform),
        )

        self.generated_images = (torch.Tensor(), torch.Tensor())
        self.test_saver = MinMaxSaver()
        self.min_anomaly_size = dataset_cls.anomaly_info(hparams)["min_size"]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z

    def training_step(self, batch: dict, batch_idx) -> dict:
        x = batch["image"]
        x_hat, z = self(x)

        crit_loss = self.criterion(x_hat, x).mean()
        tqdm_dict = {"loss/train": crit_loss}
        return OrderedDict(
            {"loss": crit_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

    def validation_step(self, batch: dict, batch_idx) -> dict:
        x = batch["image"]
        x_hat, z = self(x)

        if self.current_epoch % 20 == 19 and len(self.generated_images[0]) < 4:
            self.generated_images = (x_hat[:4].cpu(), x[:4].cpu())

        crit_loss = self.criterion(x_hat, x).mean()
        return OrderedDict({"loss/val": crit_loss})

    def validation_end(self, outputs: Iterable[dict]) -> dict:
        if len(self.generated_images[0]) > 0:
            self.logger.experiment.add_image(
                "generated_images/val",
                make_grid(
                    torch.cat(self.generated_images),
                    nrow=len(self.generated_images[0]),
                ),
                self.current_epoch,
            )
            self.generated_images = (torch.Tensor(), torch.Tensor())

        tqdm_dict = {}
        for metric in ["loss/val"]:
            tqdm_dict[metric] = sum(o[metric] for o in outputs) / len(outputs)
        return {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "loss/val": tqdm_dict["loss/val"],
        }

    def test_step(self, batch: dict, batch_idx) -> dict:
        x, target = batch["image"], batch["target"]
        x_hat, z = self(x)

        if len(self.generated_images[0]) < 8 and (target != 0).all():
            self.generated_images = (
                torch.cat(
                    [
                        self.generated_images[0],
                        x_hat[:2].cpu(),
                        x_hat[-2:].cpu(),
                    ]
                ),
                torch.cat(
                    [self.generated_images[1], x[:2].cpu(), x[-2:].cpu()]
                ),
            )

        # Only channel mean for prediction map.
        pred_map = self.criterion(x_hat, x).mean(dim=1).cpu()
        crit_loss = pred_map.mean(dim=[1, 2])  # Height & width mean.

        # Save the image if it has the lowest prediction.
        self.test_saver.update(
            x, target, crit_loss, x_hat, pred_map, batch.get("mask", None)
        )

        return OrderedDict(
            {
                "prediction": crit_loss,
                "target": target,
                "pred_map": pred_map,
                "mask": batch.get("mask", None),
                "z": z,
            }
        )

    def test_end(self, outputs: Iterable[dict]) -> dict:
        if len(self.generated_images[0]) > 0:
            self.logger.experiment.add_image(
                "generated_images/test",
                make_grid(
                    torch.cat(self.generated_images),
                    nrow=len(self.generated_images[0]),
                ),
                self.current_epoch,
            )
            self.generated_images = (torch.Tensor(), torch.Tensor())

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

        # Compute & draw roc / connected component analysis.
        pred = torch.cat([output["prediction"].cpu() for output in outputs], 0)
        target = torch.cat([output["target"].cpu() for output in outputs], 0)
        pred_map = torch.cat(
            [output["pred_map"].cpu() for output in outputs], 0
        )

        # Classification evaluation.
        weak_pred = connected_component_rocs(
            target != 0,
            pred_map.numpy(),
            min_anomaly_size=self.min_anomaly_size,
        )

        tqdm_dict = {}
        tqdm_dict["full_auc/test"] = log_roc_figure(
            self.logger, target, pred, self.current_epoch, kind="full"
        )
        tqdm_dict["weak_auc/test"] = log_roc_figure(
            self.logger, target, weak_pred, self.current_epoch, kind="weak"
        )
        print(
            "test weak auc: {} (at component size {})".format(
                tqdm_dict["weak_auc/test"], self.min_anomaly_size
            )
        )

        # Segmentation evaluation.
        seg_eval = self.testset.load_masks
        if seg_eval:
            mask = torch.cat([output["mask"].cpu() for output in outputs], 0)
            # Flatten all images so that the roc can be computed per pixel.
            tqdm_dict["seg_auc/test"] = log_roc_figure(
                self.logger,
                torch.flatten(mask) // 255,
                torch.flatten(pred_map),
                self.current_epoch,
                kind="seg",
            )
            print(
                "test segmentation auc: {}".format(tqdm_dict["seg_auc/test"])
            )

        # Latent space evaluation & visualisation.
        z = torch.cat([output["z"] for output in outputs], 0)
        tsne_fig = latent_tsne_figure(target, z)
        self.logger.experiment.add_figure(
            "latent/tsne/test", tsne_fig, self.current_epoch
        )
        pca_fig = latent_pca_figure(target, z)
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
            num_workers=2,
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
        print("Testing with {} images".format(len(self.testset)))
        return DataLoader(self.testset, batch_size=self.hparams.batch_size)

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
            "--batch_size", type=int, default=64, help="size of the batches"
        )
        parser.add_argument(
            "--loss_criterion",
            type=str,
            choices=["mse", "l1"],
            default="mse",
            help="loss criterion",
        )
        parser.add_argument(
            "--lr", type=float, default=0.001, help="adam: learning rate"
        )
        parser.add_argument(
            "--augment",
            action="store_true",
            help="Enable data augmentation or not",
        )
        parser.add_argument(
            "--pretrained", action="store_true", help="use pre-trained model"
        )
        return parser
