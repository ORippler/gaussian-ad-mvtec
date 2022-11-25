from heapq import heappush, heappop
import math

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.ndimage import filters
from skimage.measure import label

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import cv2
from typing import Optional, Callable, Sequence, Tuple
from pytorch_lightning.logging import TestTubeLogger

_COLORS_ROC = ("darkorange", "red", "green")
_COLORS_EMB = ("blue", "red", "green")


# From https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5.
def sample_covariance(
    m: torch.Tensor, rowvar: bool = True, inplace: bool = False
) -> torch.Tensor:
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def imscatter(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    imageData: torch.Tensor,
    unnormalize_fn: Optional[Callable] = None,
    zoom: int = 1,
) -> None:
    """Scatter plot with images instead of points on ax."""
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i]
        if unnormalize_fn is not None:
            img = unnormalize_fn(img)
        img *= 255.0
        img = img.permute([1, 2, 0]).numpy().astype(np.uint8)
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords="data", frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def latent_figure(
    y_true: torch.Tensor,
    z_embedded: np.ndarray,
    name: str = "",
    images: Optional[torch.Tensor] = None,
    unnormalize_fn: Optional[Callable] = None,
) -> Sequence[Figure]:
    """Visualize the embedded latent space and return the figure.

    If images is given, return a second figure with images instead of points.
    """
    fig_points, ax_points = plt.subplots()
    # Normal plot using colored points.
    for unique in np.unique(y_true):
        ax_points.plot(
            z_embedded[y_true == unique][:, 0],
            z_embedded[y_true == unique][:, 1],
            "r.",
            color=_COLORS_EMB[unique],
            figure=fig_points,
            alpha=0.3,
        )
    ax_points.set_title("Latent Space {}".format(name))

    if images is not None:
        fig_images, ax_images = plt.subplots()
        # Save figure in high resolution so it is actually viewable
        fig_images.set_dpi(fig_images.dpi * 2)
        imscatter(
            z_embedded[:, 0],
            z_embedded[:, 1],
            ax_images,
            images,
            unnormalize_fn=unnormalize_fn,
            zoom=0.08,
        )
        ax_images.set_title("Latent Space {}".format(name))
        return fig_points, fig_images
    return fig_points


def latent_tsne_figure(
    y_true: torch.Tensor,
    z: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    unnormalize_fn: Optional[Callable] = None,
) -> Sequence[Figure]:
    """Visualize the latent space using t-SNE and return the figure.

    If images is given, return a second figure with images instead of points.
    """
    z = z.cpu().numpy()
    z_embedded = TSNE(n_components=2).fit_transform(z)
    return latent_figure(
        y_true,
        z_embedded,
        name="(t-SNE)",
        images=images,
        unnormalize_fn=unnormalize_fn,
    )


def latent_pca_figure(
    y_true: torch.Tensor,
    z: torch.Tensor,
    images: Optional[torch.Tensor] = None,
    unnormalize_fn: Optional[Callable] = None,
) -> Sequence[Figure]:
    """Visualize the latent space using PCA and return the figure.

    If images is given, return a second figure with images instead of points.
    """
    z = z.cpu().numpy()
    z_embedded = PCA(n_components=2).fit_transform(z)
    return latent_figure(
        y_true,
        z_embedded,
        name="(PCA)",
        images=images,
        unnormalize_fn=unnormalize_fn,
    )


def roc_figure(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> Tuple[Figure, dict]:
    """Draw a receiver operating characteristic curve into a matplotlib figure.

    Returns (figure, auroc)
    """

    _names = ("anomalies",)

    lw = 2
    fig, axs = plt.subplots(
        nrows=2, constrained_layout=True, figsize=(6.4, 9.6)
    )
    aurocs = {}

    x_labels = ("False Positive Rate", "Recall")
    y_labels = ("True Positive Rate", "Precision")
    positions = ("lower right", "lower left")
    titles = ("ROC", "PR")
    fcts = (roc_curve, precision_recall_curve)

    for (ax, x_lbl, y_lbl, pos, title, fct) in zip(
        axs, x_labels, y_labels, positions, titles, fcts
    ):
        aurocs[title] = {}

        for unique in torch.unique(y_true):
            if unique == 0:  # use normal label to test for all anomalies
                _y_true = y_true != 0
                _y_pred = y_pred
                _pos_label = None
            else:
                if len(torch.unique(y_true)) == 2:
                    break
                _y_pred = y_pred[(y_true == unique) ^ (y_true == 0)]
                _y_true = y_true[(y_true == unique) ^ (y_true == 0)]
                _pos_label = unique.numpy()

            if fct is roc_curve:
                fpr, tpr, _ = fct(_y_true, _y_pred, pos_label=_pos_label)
            else:
                # Stupid inversion of labels by sklearn
                tpr, fpr, _ = fct(_y_true, _y_pred, pos_label=_pos_label)
            _auroc = auc(fpr, tpr)
            aurocs[title][_names[unique]] = _auroc

            ax.plot(
                fpr,
                tpr,
                color=_COLORS_ROC[unique],
                figure=fig,
                lw=lw,
                label="{} (AUC: {:0.2f})".format(_names[unique], _auroc),
            )
            if fct is roc_curve:
                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="navy",
                    lw=lw,
                    linestyle="--",
                    figure=fig,
                )
            else:
                ax.plot(
                    [0, 1],
                    [1, 0],
                    color="navy",
                    lw=lw,
                    linestyle="--",
                    figure=fig,
                )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(title)
        ax.legend(loc=pos)
    return fig, aurocs


def scatter_figure(y_true: torch.Tensor, y_pred: torch.Tensor) -> Figure:
    """Draw a scatter plot of the predictions into a matplotlib figure."""
    assert y_true.shape == y_pred.shape
    assert y_true.dim() == 1  # Otherwise boxplot will plot multiple plots.

    fig, ax = plt.subplots()

    ax.boxplot(y_pred, showfliers=False, notch=True)

    _num_unique = len(np.unique(y_true))
    _num_unique = _num_unique - 1 if _num_unique > 1 else _num_unique
    for unique in np.unique(y_true):
        _pred = y_pred[y_true == unique]
        x = np.random.normal(
            0.8 + (0.4 / _num_unique) * unique, 0.04, size=len(_pred)
        )
        ax.plot(
            x, _pred, "r.", color=_COLORS_EMB[unique], figure=fig, alpha=0.3
        )
    ax.set_ylabel("Output Prediction")
    ax.set_title("Predictions")
    return fig


def log_roc_figure(
    logger: TestTubeLogger,
    target: torch.Tensor,
    pred: torch.Tensor,
    epoch: int,
    kind: Optional[str] = None,
    data: str = "test",
) -> dict:
    """Logs a ROC curve and scatter plot figure to the given logger.

    If quantiles is given as an increasing array of boundary points,
    additionally plot a figure per anomaly size quantile (using all anomalies).

    Returns the AUROC score (globally). If quantiles is given, additionally
    returns the AUROC per quantile as second return value.
    """
    fig, auroc = roc_figure(target, pred)
    logger.experiment.add_figure(
        "{}/roc/{}".format(kind, data) if kind else "roc/{}".format(data),
        fig,
        epoch,
    )
    logger.experiment.add_figure(
        "{}/scatter/{}".format(kind, data)
        if kind
        else "scatter/{}".format(data),
        scatter_figure(target, pred),
        epoch,
    )
    return auroc


# TODO: REFACTOR
def connected_component_rocs(
    y_true: torch.Tensor, y_pred: np.ndarray, min_anomaly_size: float
) -> np.ndarray:
    """Draw a receiver operating characteristic surface into a figure.

    The receiver operating surface's second dimension is using
    connected_component_pred().

    Args:
        y_true (Tensor(N)): ground truth of the samples.
        y_pred (array(NxHxW)): prediction map for each sample.
        min_anomaly_size (float): minimal size of an anomaly (as fraction of
            the image area).

    Returns weak_pred
    """
    roc_resolution = 1000  # Resolution of the x & y axis in the surface plot.
    img_area = y_pred.shape[1] * y_pred.shape[2]
    component_sizes = np.geomspace(1, img_area, num=roc_resolution, dtype=int)

    # Add the precise min_anomaly_size (to be exact).
    min_anomaly_pixels = int(min_anomaly_size * img_area)
    # Actually we could insert directly with log(min_anomaly_pixels) :/.
    # let's do a quick binary search
    min_size_index = np.searchsorted(component_sizes, min_anomaly_pixels)
    # Insert at the correct position.
    component_sizes = np.insert(
        component_sizes, min_size_index, min_anomaly_pixels
    )

    # Aggregate all thresholds in a num_samples x component_size array
    thresholds = np.stack(
        [
            connected_component_pred(pred_image, component_sizes)
            for pred_image in y_pred
        ]
    )

    # Transpose to get component_size x num_samples.
    thresholds = thresholds.transpose()

    if min_anomaly_size is not None:
        # Extract the precise weakly supervised auroc again.
        weak_pred = thresholds[min_size_index]
        thresholds = np.delete(thresholds, min_size_index, axis=0)
        component_sizes = np.delete(component_sizes, min_size_index, axis=0)

    return weak_pred


def connected_component_pred(
    y_pred: np.ndarray,
    component_sizes: np.ndarray,
    connectivity: int = 2,
    resolution: int = 1000,
) -> np.ndarray:
    """Compute maximum thresholds on y_pred to contain connected components.

    y_pred then contains components of at least component_size pixels.

    Args:
        y_pred (np.float(h x w)): prediction map (anomaly score per pixel).
        component_sizes (np.array(int)): Minimum component sizes.
        connectivity (int): Maximum number of orthogonal hops to consider a
            pixel/voxel as neighbor.
            Accepted values are ranging from 1 to y_pred.ndim.
            If None, a full connectivity of y_pred.ndim is used.
        resolution (int): Number of possible threshold values to consider
            (linearly spaces between min and max).

    Returns:
        thresholds (np.array(float)): Thresholds on the predicted values
            matching component_sizes.
    """
    # Sort and unique-ify all pixel values to process in order.
    min_val, max_val = y_pred.min(), y_pred.max()
    values = np.linspace(min_val, max_val, num=resolution)

    # Output array of thresholds per component_size.
    thresholds = np.empty_like(component_sizes, dtype=y_pred.dtype)
    size_index = 0
    for value in reversed(values):  # reversed order to go from high to low.
        arr = (y_pred >= value).astype(np.uint8, copy=False)
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            arr, connectivity=connectivity * 4
        )

        # biggest connected component is number of entries (excluding 0)
        max_count = stats[1:, cv2.CC_STAT_AREA].max()
        while component_sizes[size_index] <= max_count:
            thresholds[size_index] = value
            size_index += 1
            if size_index == len(component_sizes):
                return thresholds
    raise ValueError(
        "Component size unreachable: {}".format(component_sizes[size_index])
    )


class MinMaxSaver:
    """Save min/max predicted images for good&anomaly images during testing."""

    def __init__(self, unnormalize_fn: Optional[Callable] = None) -> None:
        self.min_good_pred, self.max_good_pred = None, None
        self.min_anomaly_pred, self.max_anomaly_pred = None, None
        self.unnormalize_fn = unnormalize_fn

    def update(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        pred: torch.Tensor,
        *args: torch.Tensor
    ) -> None:
        """
        Store all relevant 2D-arrays in lists
        """
        args = [arg for arg in args if arg is not None]

        def _adapt_pred_slice(pred_slice: torch.Tensor) -> torch.Tensor:
            # Perform necessary dimension wrangling and type castings for
            # make_grid().
            pred_slice = pred_slice.cpu()
            # Convert segmentation_mask to 0-1 range for tensorboard.
            if pred_slice.dtype == torch.uint8:
                pred_slice = pred_slice.float()
                pred_slice = pred_slice / pred_slice.max()
            if pred_slice.ndim == 2:
                num_stacks = x.size(1)
                return torch.stack([pred_slice] * num_stacks)
            else:
                return pred_slice

        min_good_pred = (
            pred[target == 0].min() if (target == 0).any() else np.inf
        )
        max_good_pred = (
            pred[target == 0].max() if (target == 0).any() else -np.inf
        )

        if (target == 0).any():
            if (
                self.min_good_pred is None
                or self.min_good_pred[0] > min_good_pred
            ):
                self.min_good_pred = [
                    min_good_pred,
                    x[target == 0][pred[target == 0].argmin()].cpu(),
                ]
                for arg in args:
                    _slice = arg[target == 0][pred[target == 0].argmin()]
                    self.min_good_pred.append(_adapt_pred_slice(_slice))

            if (
                self.max_good_pred is None
                or self.max_good_pred[0] < max_good_pred
            ):
                self.max_good_pred = [
                    max_good_pred,
                    x[target == 0][pred[target == 0].argmax()].cpu(),
                ]
                for arg in args:
                    _slice = arg[target == 0][pred[target == 0].argmax()]
                    self.max_good_pred.append(_adapt_pred_slice(_slice))

        min_anomaly_pred = (
            pred[target != 0].min() if (target != 0).any() else np.inf
        )
        max_anomaly_pred = (
            pred[target != 0].max() if (target != 0).any() else -np.inf
        )
        # TODO: Adapt to store images per anomaly class
        if (target != 0).any():
            if (
                self.min_anomaly_pred is None
                or self.min_anomaly_pred[0] > min_anomaly_pred
            ):
                self.min_anomaly_pred = [
                    min_anomaly_pred,
                    x[target != 0][pred[target != 0].argmin()].cpu(),
                ]
                for arg in args:
                    _slice = arg[target != 0][pred[target != 0].argmin()]
                    self.min_anomaly_pred.append(_adapt_pred_slice(_slice))

            if (
                self.max_anomaly_pred is None
                or self.max_anomaly_pred[0] < max_anomaly_pred
            ):
                self.max_anomaly_pred = [
                    max_anomaly_pred,
                    x[target != 0][pred[target != 0].argmax()].cpu(),
                ]
                for arg in args:
                    _slice = arg[target != 0][pred[target != 0].argmax()]
                    self.max_anomaly_pred.append(_adapt_pred_slice(_slice))

    def good_grid(self) -> torch.Tensor:
        min_good = self.min_good_pred[1:]
        max_good = self.max_good_pred[1:]
        if self.unnormalize_fn is not None:
            # TODO: replace image only normalization with flags/better solution
            # if desired in future.
            min_good[0], max_good[0] = (
                self.unnormalize_fn(min_good[0]),
                self.unnormalize_fn(max_good[0]),
            )
        grid_list = min_good + max_good
        return make_grid(torch.stack(grid_list), nrow=len(min_good))

    def anomaly_grid(self) -> torch.Tensor:
        min_anomaly = self.min_anomaly_pred[1:]
        max_anomaly = self.max_anomaly_pred[1:]
        if self.unnormalize_fn is not None:
            # TODO: replace image only normalization with flags/better solution
            # if desired in future.
            min_anomaly[0] = self.unnormalize_fn(min_anomaly[0])
            max_anomaly[0] = self.unnormalize_fn(max_anomaly[0])
        grid_list = min_anomaly + max_anomaly
        return make_grid(torch.stack(grid_list), nrow=len(min_anomaly))
