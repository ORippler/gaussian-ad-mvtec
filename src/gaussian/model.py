import argparse
from collections import OrderedDict
import copy

from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

import pytorch_lightning as pl

from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.svm import OneClassSVM

from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import make_grid

from ..common.augmentation import (
    dataset_aug,
    Compose,
    unnormalize,
    MaskToSize,
)
from albumentations import Resize, Normalize

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
    sample_covariance,
)
from ..common.utils import TensorList, batched_index_select, flatten
from .transparent import build_transparent, MODEL_NAMES
from tqdm import tqdm
from typing import NoReturn, Union, Iterable, Optional, Type


class Mahalanobis(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        dataset_cls: Type[AnomalyDetectionDataset],
    ) -> NoReturn:
        super(Mahalanobis, self).__init__()

        self.hparams = hparams

        self.normalization_epochs = 100 if self.hparams.augment else 1

        self.model, self.input_size = build_transparent(
            hparams.arch,
            pretrained=True,
            extract_blocks=hparams.extract_blocks,
            freeze=True,
        )

        if hasattr(self.model, "fc"):
            self.model.fc = nn.Identity()

        # We do not want to augment in val/test
        final_transform = [Normalize(), ToTensorV2()]
        final_eval_transform = copy.deepcopy(final_transform)

        if hparams.augment:
            uncached = Compose(
                [dataset_aug(hparams, dataset_cls), *final_transform]
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
            transform=Compose(
                [
                    Resize(self.input_size, self.input_size),
                    # Reduce mask to size for max perf.
                    MaskToSize(),
                ]
            ),
            uncached_transform=Compose(final_eval_transform),
        )

        self.test_saver = MinMaxSaver(unnormalize_fn=unnormalize)

        if self.hparams.variance_threshold:
            assert (
                self.hparams.pca != self.hparams.npca
            ), "need to set either only pca or npca flag"
        else:
            assert (
                not self.hparams.pca and not self.hparams.npca
            ), "cannot specify pca/npca and not specify variance_threshold"

        self.statistics_computed = False

        self._device = torch.device(
            "cuda"
            if torch.cuda.is_available() and self.hparams.gpus
            else "cpu"
        )

    def forward(self, x: torch.Tensor) -> TensorList:
        """Output features of self.model."""
        with torch.no_grad():
            z = self.model(x)
        return z

    @staticmethod
    def tensorlist_or_tensor(items: list) -> Union[torch.Tensor, TensorList]:
        if len(items) == 1:
            return items[0].unsqueeze(0)
        return TensorList(items)

    def compute_train_sed(self, features: TensorList) -> NoReturn:
        """Compute sed normalization mean & stddev.

        This is the per feature independent gaussian assumption (only mean
        and stddev).
        """

        mean = features.mean(dim=1)  # mean is level x features.
        stddev = features.std(dim=1)
        self.sed_mean = mean  # equal to the mean of mvg, but not equal to pca_mean (as sed/maha may be performed on reduced dimensionalities)
        self.sed_stddev = stddev

    @staticmethod
    def compute_mahalanobis_threshold(
        k: int, p: float = 0.9973
    ) -> torch.Tensor:
        """Compute a threshold on the mahalanobis distance.

        So that the probability of mahalanobis with k dimensions being less
        than the returned threshold is p.
        """
        # Mahalanobis² is Chi² distributed with k degrees of freedom.
        # So t is square root of the inverse cdf at p.
        return torch.Tensor([chi2.ppf(p, k)]).sqrt()

    def compute_train_gaussian(self, features: TensorList) -> NoReturn:
        """
        features: TensorList
        """

        def fit_inv_covariance(samples):
            return torch.Tensor(LedoitWolf().fit(samples.cpu()).precision_).to(
                samples.device
            )

        print("Performing Covariance Estimation")
        inv_covariance = TensorList(
            [fit_inv_covariance(level) for level in features]
        )
        mean = features.mean(dim=1)  # mean features.

        self.mvg_mean = mean
        self.mvg_inv_covariance = inv_covariance
        # Also cache the number of features each level outputs for later.
        feature_count = torch.cat(
            [torch.Tensor([level.shape[-1]]) for level in mean]
        )
        self.feature_count = feature_count

    def compute_train_ocsvm(self, features: TensorList) -> NoReturn:
        def fit_ocsvm(samples):
            ocsvm = OneClassSVM(kernel="rbf", gamma="scale")
            ocsvm.fit(samples)
            return ocsvm

        print("Fitting OCSVM")
        self.ocsvms = [fit_ocsvm(level) for level in features.cpu()]

    def ocsvm_predict(self, features: TensorList) -> TensorList:
        return TensorList(
            [
                torch.Tensor(
                    -ocsvm.decision_function(level.mean(dim=(-2, -1)).cpu())
                )
                for level, ocsvm in zip(features, self.ocsvms)
            ]
        )

    def compute_pca(
        self, features: TensorList, variance_threshold: float = 0.95
    ) -> NoReturn:
        """Compute pca normalization of teacher features retaining variance.

        Contrary to normal pca, this throws away the features with large
        variance and only keeps the ones with small amounts of variance.
        It is expected that those features will activate on the anomalies.
        """

        mean = features.mean(dim=1)  # mean is level x features.

        def fit_level(features: torch.Tensor) -> torch.Tensor:
            pca = PCA(n_components=None).fit(features)
            # Select features above variance_threshold.
            variances = pca.explained_variance_ratio_.cumsum()
            last_component = (variances > variance_threshold).argmax()
            # last_component is the index of the last value needed to reach at
            # least the required explained variance.
            # As the true variance lies somewhere in between [last_component - 1,
            # last_component], we include the whole interval for both pca as
            # well as NPCA based dimensionality reduction
            if self.hparams.pca:
                return torch.Tensor(pca.components_[: last_component + 1])
            elif self.hparams.npca:
                return torch.Tensor(pca.components_[last_component - 1 :])
            else:
                raise ValueError(
                    "either hparams.pca or hparams.npca need to be specified"
                )

        components = self.tensorlist_or_tensor(
            [fit_level(level) for level in features.cpu()]
        )
        self.pca_mean = mean
        self.pca_components = components

    # TensorList if len(extract_blocks) > 1 else torch.Tensor
    def l2_distance(
        self, features: Union[torch.Tensor, TensorList]
    ) -> Union[torch.Tensor, TensorList]:
        return (
            (features - self.mvg_mean.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))
            .mean(dim=(-2, -1))
            .pow(2)
            .mean(dim=2)
            .sqrt()
        )

    def sed_distance(
        self, features: Union[torch.Tensor, TensorList]
    ) -> Union[torch.Tensor, TensorList]:
        """Return normalized features (using the computed normalization)."""
        # Unsqueeze batch, height & width.
        return (
            (
                (
                    features
                    - self.sed_mean.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )
                / self.sed_stddev.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            )
            .mean(dim=(-2, -1))
            .pow(2)
            .mean(dim=2)
            .sqrt()
        )

    def pca_reduction(
        self, features: Union[torch.Tensor, TensorList]
    ) -> Union[torch.Tensor, TensorList]:
        """Return pca-reduced features (using the computed PCA)."""
        # Features is level x training_samples x features x height x width.
        # Unsqueeze batch, height & width.
        demeaned = features - self.pca_mean.unsqueeze(1).unsqueeze(
            -1
        ).unsqueeze(-1)

        def batched_mul_components(
            level: torch.Tensor, components: torch.Tensor
        ) -> torch.Tensor:
            # Cannot use einsum because of unsupported broadcasting.
            # So do a permute to (samples x height x width x features).
            reduced = torch.matmul(  # Batched vector matrix multiply.
                level.permute(0, 2, 3, 1).unsqueeze(-2),
                components.t().unsqueeze(0).unsqueeze(0).unsqueeze(0),
            ).squeeze(
                -2
            )  # Squeeze so this is vector matrix multiply.
            return reduced.permute(0, 3, 1, 2)  # Back to BCHW.

        return self.tensorlist_or_tensor(
            # This is (X - mean).dot(components.t()).
            [
                batched_mul_components(level, components)
                for level, components in zip(
                    demeaned, self.pca_components.to(self._device)
                )
            ]
        )

    @staticmethod
    def mahalanobis_distance(
        values: torch.Tensor, mean: torch.Tensor, inv_covariance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the batched mahalanobis distance.

        values is a batch of feature vectors.
        mean is either the mean of the distribution to compare, or a second
        batch of feature vectors.
        inv_covariance is the inverse covariance of the target distribution.
        """
        assert values.dim() == 2
        assert 1 <= mean.dim() <= 2
        assert inv_covariance.dim() == 2
        assert values.shape[1] == mean.shape[-1]
        assert mean.shape[-1] == inv_covariance.shape[0]
        assert inv_covariance.shape[0] == inv_covariance.shape[1]

        if mean.dim() == 1:  # Distribution mean.
            mean = mean.unsqueeze(0)
        x_mu = values - mean  # batch x features
        # Same as dist = x_mu.t() * inv_covariance * x_mu batch wise
        dist = torch.einsum("im,mn,in->i", x_mu, inv_covariance, x_mu)
        return dist.sqrt()

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return OrderedDict({"loss/val": torch.tensor(0)})

    def validation_end(self, outputs: list) -> dict:
        return {"progress_bar": {}, "log": {}}

    def compute_train_statistics(self) -> NoReturn:
        # Use same dataloader as training.
        trainset = self.datasplit.train()
        print(
            "Computing training statistics with {} images".format(
                len(trainset)
            )
        )
        dataloader = DataLoader(
            trainset, batch_size=self.hparams.batch_size, num_workers=2
        )
        with torch.no_grad():
            outputs = []
            for epoch in tqdm(range(self.normalization_epochs)):
                for batch in dataloader:
                    z = self.model(batch["image"].to(self._device))
                    # Mean across locations (at this point to save GPU RAM).
                    z = z.mean(dim=(3, 4), keepdim=True)
                    outputs.append(z)
            # Features is level x training_samples x features.
            features = TensorList.cat(outputs, dim=1)

        # TODO: Beautify this so pca reduction does not require singleton dimensions for hxw
        if self.hparams.npca or self.hparams.pca:
            self.compute_pca(
                features.mean(dim=(3, 4)),
                variance_threshold=self.hparams.variance_threshold,
            )

            outputs_reduced = []
            for batch in outputs:
                reduced = self.pca_reduction(batch)
                outputs_reduced.append(reduced.mean(dim=(3, 4)))
            features = TensorList.cat(outputs_reduced, dim=1)
        else:
            features = features.mean(dim=(3, 4))

        self.compute_train_gaussian(features)
        if self.hparams.ocsvm:
            self.compute_train_ocsvm(features)
        if self.hparams.sed:
            self.compute_train_sed(features)

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        x, target, mask = batch["image"], batch["target"], batch["mask"]
        z = self(x)

        if not self.statistics_computed:
            self.compute_train_statistics()
            self.statistics_computed = True

        # reduce features also here
        if self.hparams.npca or self.hparams.pca:
            z = self.pca_reduction(z)

        maha = TensorList(
            [
                self.mahalanobis_distance(
                    level.mean(dim=(-2, -1)), val_mean, val_icov
                )
                for level, val_mean, val_icov in zip(
                    z, self.mvg_mean, self.mvg_inv_covariance
                )
            ]
        )

        if self.hparams.ocsvm:
            ocsvm = self.ocsvm_predict(z)

        if self.hparams.sed:
            sed = self.sed_distance(z)

        if self.hparams.l2:
            l2 = self.l2_distance(z)

        self.test_saver.update(x, target, maha.mean(dim=0))

        result = OrderedDict({"target": target, "x": x.cpu(), "mask": mask})
        for name in ("maha", "sed", "l2", "ocsvm"):
            if name in locals():
                result[name] = locals()[name].cpu()

        return result

    def evaluate_latent(
        self,
        outputs: Iterable[dict],
        target: torch.Tensor,
        key: str,
        images: Optional[torch.Tensor] = None,
    ) -> dict:
        """Plot & evaluate the latent space from test output.

        Return the new entries to the log dict as a dict.
        """
        # Latent space is given as [batch x dimension] tensors for each level.
        z = TensorList.cat([output[key] for output in outputs], dim=1)
        # Iterate all levels of the latent space.
        result = {}
        for i, (z_i, feature_c) in enumerate(zip(z, self.feature_count)):
            # Map index to correct attach_block level.
            i = self.hparams.extract_blocks[i]
            pred = z_i

            auc = log_roc_figure(
                self.logger,
                target,
                pred,
                self.current_epoch,
                kind="latent_{}/level_{}".format(key, i),
            )
            result["latent_{}/level_{}/auc/test".format(key, i)] = auc

            if key == "maha":
                # Mahalanobis TPR/FPR at n sigma.
                sigmas = [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ]  # values > 8 are not computable.
                for sigma in sigmas:
                    # Probability of a Gaussian at n sigma.
                    p = torch.erf(sigma / torch.DoubleTensor([2]).sqrt())
                    # Threshold on mahalanobis distance at p (n sigma).
                    threshold = self.compute_mahalanobis_threshold(
                        feature_c, p=p.item()
                    )
                    anomaly_pred = pred > threshold
                    tpr = (
                        anomaly_pred[target != 0].sum()
                        / (target != 0).sum().float()
                    )
                    result[
                        "latent_{}/level_{}/sigma_{}/tpr/test".format(
                            key, i, sigma
                        )
                    ] = tpr
                    fpr = (
                        anomaly_pred[target == 0].sum()
                        / (target == 0).sum().float()
                    )
                    result[
                        "latent_{}/level_{}/sigma_{}/fpr/test".format(
                            key, i, sigma
                        )
                    ] = fpr

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

        # compute & draw roc / connected component analysis
        target = torch.cat([output["target"].cpu() for output in outputs], 0)
        x = torch.cat([output["x"] for output in outputs], 0)

        tqdm_dict = {}
        # Classification evaluation.

        for distance in ("maha", "sed", "l2", "ocsvm"):
            # check if distance is in returned dict
            if distance in outputs[0]:
                pred = torch.cat(
                    [
                        output.get(distance).mean(dim=0).cpu()
                        for output in outputs
                    ],
                    0,
                )

                tqdm_dict[
                    "{}/full_auc/test".format(distance)
                ] = log_roc_figure(
                    self.logger,
                    target,
                    pred,
                    self.current_epoch,
                    kind="full/{}".format(distance),
                )
                print(
                    "{} test auc: {}".format(
                        distance,
                        tqdm_dict.get("{}/full_auc/test".format(distance)),
                    )
                )

                # Latent space evaluation / visualisation.
                tqdm_dict.update(
                    self.evaluate_latent(outputs, target, distance, images=x)
                )

        if self.hparams.npca:
            # Log how many features where removed.
            for i, level in enumerate(self.pca_components):
                # Map index to correct attach_block level.
                i = self.hparams.extract_blocks[i]
                percentage = level.shape[0] / level.shape[1]
                print(
                    "npca features level {}: {} ({})".format(
                        i, level.shape[0], percentage
                    )
                )
                tqdm_dict["npca_feat/level_{}/test".format(i)] = level.shape[0]
                tqdm_dict["npca_perc/level_{}/test".format(i)] = percentage
        tqdm_dict = flatten(tqdm_dict)
        return {"progress_bar": tqdm_dict, "log": tqdm_dict}

    # required by pytorch lightning
    def configure_optimizers(self) -> list:
        return []

    @pl.data_loader
    def train_dataloader(self) -> DataLoader:
        # already shuffled in datasplit
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
        """Specify the hyperparams for this LightningModule."""
        # MODEL specific
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
            "--extract_blocks",
            type=int,
            nargs="+",
            default=[5],
            help="Blocks to extract the features from "
            "Compared to the paper, we have an index"
            " offset of 1 (we star 0 based here but"
            " with base 1 in the paper)",
        )
        parser.add_argument(
            "--batch_size", type=int, default=64, help="size of the batches"
        )
        parser.add_argument(
            "--variance_threshold",
            type=float,
            default=None,
            help="variance threshold to apply",
        )
        parser.add_argument("--pca", action="store_true", help="enable pca")
        parser.add_argument("--npca", action="store_true", help="Enable npca")
        parser.add_argument(
            "--augment", action="store_true", help="Enable data augmentation"
        )
        parser.add_argument(
            "--l2", action="store_true", help="Evaluate l2 distance"
        )
        parser.add_argument("--sed", action="store_true", help="SED distance")
        parser.add_argument("--ocsvm", action="store_true", help="train ocsvm")
        return parser
