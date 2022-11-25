import torch
from torch import nn
import torchvision.models as models
from typing import Tuple

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


class FeatureExtractorResnet(nn.Module):
    """Wraps a resnet for extracting last features as well as fine-tuning
    classification."""

    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.fc(features)

        return features, x


class FeatureExtractorEfficientnet(nn.Module):
    """Wraps a efficientnet for extracting last features as well as fine-tuning
    classification.
    """

    def __init__(self, efficientnet: nn.Module) -> None:
        super().__init__()
        # Keep around the original net but override the forward call.
        self.net = efficientnet

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calls extract_features to extract features and returns logits."""
        bs = inputs.size(0)
        # Convolution layers.
        x = self.net.extract_features(inputs)

        # Pooling and final linear layer.
        x = self.net._avg_pooling(x)
        x = x.view(bs, -1)
        features = self.net._dropout(x)
        x = self.net._fc(x)
        return features, x


def initialize_model(
    model_name: str, num_classes: int, use_pretrained: bool = True
) -> Tuple[nn.Module, int]:
    model_ft = None
    input_size = 0
    if model_name.startswith("resnet"):
        """Resnet18"""
        model_ft = models.__dict__[model_name](pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        # Wrap a FeatureExtractorResnet around this so that it gives the
        # feature output as well.
        model_ft = FeatureExtractorResnet(model_ft)

    elif model_name.startswith("efficientnet"):
        """ EfficientNet"""
        from efficientnet_pytorch import EfficientNet
        from efficientnet_pytorch.utils import (
            get_same_padding_conv2d,
            round_filters,
        )

        if num_classes is None:
            num_classes = 1000  # Matching EfficientNet default.
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained(
                model_name, num_classes=num_classes
            )
        else:
            model_ft = EfficientNet.from_name(
                model_name, override_params={"num_classes": num_classes}
            )
        input_size = EfficientNet.get_image_size(model_name)
        model_ft = FeatureExtractorEfficientnet(model_ft)

    else:
        raise ValueError("Invalid model name")
    return model_ft, input_size
