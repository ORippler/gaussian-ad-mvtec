"""A general-purpose and extensible fully-convolutional auto-encoder framework.

It can load pretrained models for its encoder.
"""

import torch
import torch.nn as nn
import torchvision.models as models

import pytorch_lightning as pl
from typing import NoReturn, Optional, Union, Tuple
from collections.abc import Iterable

# pull out resnet names from torchvision models
MODEL_NAMES = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
)


class ResnetEncoder(nn.Module):
    """Wraps a ResNet for an autoencoder."""

    def __init__(self, resnet: nn.Module, latent_dim: int = 32) -> NoReturn:
        super().__init__()
        self.latent_dim = latent_dim
        # Forget about the fc layers, but copy out everything else.
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.poolconv = nn.Conv2d(512, latent_dim, kernel_size=7, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.poolconv(x)

        x = torch.flatten(x, start_dim=1)
        return x


def tconv7x7(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    padding: int = 3,
) -> nn.Module:
    """7x7 transpose convolution with padding.

    Uses PixelShuffle convolutions instead of strided convolutions.
    see Shi et al.: Is the deconvolution layer the same as a convolutional
    layer?
    """
    if stride == 1:
        return nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=7,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                out_planes * stride * stride,
                kernel_size=7,
                stride=1,
                padding=padding,
                groups=groups,
                bias=False,
                dilation=dilation,
            ),
            nn.PixelShuffle(stride),
        )


def tconv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Module:
    """3x3 transpose convolution with padding.

    Uses PixelShuffle convolutions instead of strided convolutions.
    see Shi et al.: Is the deconvolution layer the same as a convolutional
    layer?
    """
    if stride == 1:
        return nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                out_planes * stride * stride,
                kernel_size=3,
                stride=1,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation,
            ),
            nn.PixelShuffle(stride),
        )


def tconv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Module:
    """1x1 transpose convolution.

    Uses PixelShuffle convolutions instead of strided convolutions.
    see Shi et al.: Is the deconvolution layer the same as a convolutional
    layer?
    """
    if stride == 1:
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=1, stride=stride, bias=False
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                out_planes * stride * stride,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.PixelShuffle(stride),
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        planes: int,
        outplanes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None,
    ) -> NoReturn:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "BasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock"
            )
        # Both conv1 and upsample layers upsample the input when stride != 1.
        self.conv2 = tconv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv1 = tconv3x3(planes, outplanes, stride)
        self.bn1 = norm_layer(outplanes)
        self.relu = nn.ReLU()
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv2(x)
        out = self.bn2(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResnetDecoder(nn.Module):
    def __init__(
        self,
        block: Union[BasicBlock, nn.Module],
        layers: Iterable,
        latent_dim: int = 32,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Iterable] = None,
        norm_layer: Optional[nn.Module] = None,
    ) -> NoReturn:
        super(ResnetDecoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.latent_dim = latent_dim
        self.inplanes = 512
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group

        self.upconv = tconv7x7(latent_dim, self.inplanes, padding=0)
        self.relu = nn.ReLU()
        self.layer4 = self._make_layer(
            block,
            256,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.layer3 = self._make_layer(
            block,
            128,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer2 = self._make_layer(
            block,
            64,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = tconv7x7(self.inplanes, 3, stride=2)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                else:
                    raise NotImplementedError()

    def _make_layer(
        self,
        block: Union[BasicBlock, nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation //= stride
            stride = 1
        if stride != 1 or planes != self.inplanes // block.expansion:
            upsample = nn.Sequential(
                tconv1x1(self.inplanes // block.expansion, planes, stride),
                norm_layer(planes),
            )
        layers = []
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    self.inplanes // block.expansion,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                upsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes // block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = torch.reshape(x, (-1, self.latent_dim, 1, 1))
        x = self.upconv(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.upsample(x)
        x = self.conv1(x)

        x = self.sigmoid(x)

        return x


def build_encoder(
    model_name: str, pretrained: bool = True, latent_dim: int = 32
) -> nn.Module:
    """Build an encoder model, possibly pretrained."""
    if model_name.startswith("resnet"):
        model_ft = models.__dict__[model_name](pretrained=pretrained)
        input_size = 224
        return ResnetEncoder(model_ft, latent_dim=latent_dim), input_size
    else:
        raise NotImplementedError()


def build_decoder(
    model_name: str, latent_dim: int = 32
) -> Tuple[nn.Module, int]:
    """Build a decoder model."""
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model_ft = ResnetDecoder(
                BasicBlock, [2, 2, 2, 2], latent_dim=latent_dim
            )
        elif model_name == "resnet34":
            model_ft = ResnetDecoder(
                BasicBlock, [3, 4, 6, 3], latent_dim=latent_dim
            )
        else:
            raise NotImplementedError()
        output_size = 224
        return model_ft, output_size
    else:
        raise NotImplementedError()


def build_autoencoder(
    arch: str = "resnet18", pretrained: bool = True, latent_dim: int = 32
) -> Tuple[nn.Module, nn.Module, int]:
    encoder, input_size = build_encoder(
        model_name=arch, pretrained=pretrained, latent_dim=latent_dim
    )
    decoder, output_size = build_decoder(
        model_name=arch, latent_dim=latent_dim
    )
    if input_size != output_size:
        raise ValueError(
            "input_size != output_size: {} != {}".format(
                input_size, output_size
            )
        )

    return encoder, decoder, input_size
