"""Models which expose their intermediate activations in forward()."""

from copy import deepcopy
import math

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import (
    get_same_padding_conv2d,
    round_filters,
    MemoryEfficientSwish,
    Swish,
    round_repeats,
)

import torch
from torch import nn
import torchvision.models as models

from ..common.utils import TensorList
from typing import NoReturn, Iterable, Tuple, Optional

# pull out resnet names from torchvision models
MODEL_NAMES = sorted(
    [
        name
        for name in models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(models.__dict__[name])
    ]
    + ["efficientnet-b{}".format(b) for b in range(8)]
)


class TransparentResNet(nn.Module):
    """Wraps a ResNet for extracting features at multiple scales."""

    def __init__(
        self,
        resnet: nn.Module,
        extract_blocks: Iterable[int] = [5],
        freeze: bool = False,
    ) -> NoReturn:
        super().__init__()
        self.freeze = freeze
        # Forget about the fc layers, but copy out everything else.
        self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Attach blocks only as needed.
        assert len(extract_blocks) == len(
            set(extract_blocks)
        ), "extract_blocks must only contain unique indices"
        assert (
            sorted(extract_blocks) == extract_blocks
        ), "attach_block must be sorted in increasing order"
        assert set(extract_blocks) <= set(
            range(5)
        ), "only block indices from 0 to 4 are valid"

        self.extract_blocks = extract_blocks
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # freeze learnable model parameters
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> TensorList:
        if self.freeze:
            # ensure we don't update running means of normalization layers
            self.eval()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        result = []
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        for i in sorted(self.extract_blocks):
            result.append(locals()["x" + str(i)])

        if len(result) == 1:
            # Replace with torch tensor, no TensorList needed.
            return torch.stack(result, dim=0)
        return TensorList(result)


class TransparentEfficientNet(nn.Module):
    """Wraps an EfficientNet for extracting features at multiple scales."""

    def __init__(
        self,
        efficientnet: nn.Module,
        input_size: int,
        extract_blocks: Iterable[int] = [5],
        freeze: bool = False,
    ) -> NoReturn:
        super().__init__()
        self.freeze = freeze
        self.input_size = input_size
        self._global_params = efficientnet._global_params

        # Forget about the fc layers, but copy out everything else.
        self._conv_stem = efficientnet._conv_stem
        self._bn0 = efficientnet._bn0
        self._swish = efficientnet._swish

        # EfficientNet saves all layers (including repeats) in a flat _blocks
        # parameter. To keep the model comparable we keep the same approach
        # but map to extract_blocks when needed.
        self.block_to_layer = [-1]  # block index to layer index (w/o first).
        self.layer_to_block = {}  # Probably a bit more efficient forward pass.
        layer = -1
        for i, block_args in enumerate(efficientnet._blocks_args):
            block_args = block_args._replace(
                num_repeat=round_repeats(
                    block_args.num_repeat, self._global_params
                )
            )
            layer += block_args.num_repeat
            self.block_to_layer.append(layer)
            self.layer_to_block[layer] = i + 1
        assert layer == len(efficientnet._blocks) - 1  # Last layer.

        # Attach blocks only as needed.
        assert len(extract_blocks) == len(
            set(extract_blocks)
        ), "extract_blocks must only contain unique indices"
        assert (
            sorted(extract_blocks) == extract_blocks
        ), "attach_block must be sorted in increasing order"
        # Last two blocks are conv & fc layer
        assert set(extract_blocks) <= set(
            range(len(self.block_to_layer) + 1)
        ), "only block indices from 0 to {} are valid".format(
            len(self.block_to_layer)
        )
        self.extract_blocks = extract_blocks
        max_block = max(extract_blocks)
        if max_block < len(self.block_to_layer) - 1:
            # Last layer inclusive.
            self._blocks = efficientnet._blocks[
                : self.block_to_layer[max_block] + 1
            ]
        else:
            self._blocks = efficientnet._blocks
        self.orig_len_blocks = len(efficientnet._blocks)  # For drop connect.

        if max_block >= len(self.block_to_layer):
            self._conv_head = efficientnet._conv_head
            self._bn1 = efficientnet._bn1

        if max_block == len(self.block_to_layer) + 1:
            self._avg_pooling = efficientnet._avg_pooling
            self._dropout = efficientnet._dropout
            self.fc = efficientnet._fc

        # freeze learnable model parameters
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def set_swish(self, memory_efficient: bool = True) -> NoReturn:
        """Sets swish function as memory efficient (training) or standard."""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Iterable[torch.Tensor]]:
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        result = []
        if 0 in self.extract_blocks:
            result.append(x)
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / self.orig_len_blocks
            x = block(x, drop_connect_rate=drop_connect_rate)
            if (
                idx in self.layer_to_block
                and self.layer_to_block[idx] in self.extract_blocks
            ):
                result.append(x)

        # Head
        if hasattr(self, "_conv_head"):
            x = self._swish(self._bn1(self._conv_head(x)))
        if len(self.block_to_layer) in self.extract_blocks:
            result.append(x)
        return x, result

    def forward(self, inputs: torch.Tensor) -> TensorList:
        """Extracts features, applies final linear layer & returns logits."""
        if self.freeze:
            # ensure we don't update running means of normalization layers
            self.eval()
        bs = inputs.size(0)
        # Convolution layers
        x, result = self.extract_features(inputs)

        # Pooling and final linear layer
        if hasattr(self, "fc"):
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self.fc(x)
            # Ensure all are 3 dimensional latents.
            x = x.unsqueeze(-1).unsqueeze(-1)
            result.append(x)

        if len(result) == 1:
            # Replace with torch tensor, no TensorList needed.
            return torch.stack(result, dim=0)
        return TensorList(result)


def build_transparent(
    model_name: str,
    pretrained: bool = False,
    extract_blocks: Iterable[int] = [5],
    num_classes: Optional[int] = None,
    freeze: bool = False,
):
    """Build an encoder model, possibly pretrained."""
    if model_name.startswith("resnet"):
        model_ft = models.__dict__[model_name](pretrained=pretrained)
        if num_classes is not None:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        return (
            TransparentResNet(
                model_ft, extract_blocks=extract_blocks, freeze=freeze
            ),
            input_size,
        )
    elif model_name.startswith("efficientnet"):
        if num_classes is None:
            num_classes = 1000  # Matching EfficientNet default.
        if pretrained:
            model_ft = EfficientNet.from_pretrained(
                model_name, num_classes=num_classes
            )
        else:
            model_ft = EfficientNet.from_name(
                model_name, override_params={"num_classes": num_classes}
            )
        input_size = EfficientNet.get_image_size(model_name)
        return (
            TransparentEfficientNet(
                model_ft,
                input_size,
                extract_blocks=extract_blocks,
                freeze=freeze,
            ),
            input_size,
        )
    else:
        raise NotImplementedError("Unknown model name {}".format(model_name))
