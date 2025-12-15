# FPN with ResNet/ResNeXt encoders - compatible with segmentation_models_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.models as models


# FPN block with lateral connection and top-down pathway
class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels: int, skip_channels: int):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            skip = self.skip_conv(skip)
            x = x + skip
        return x


# Segmentation block for each FPN level
class SegmentationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_upsamples: int = 0):
        super().__init__()

        blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if n_upsamples > 0:
            for _ in range(n_upsamples):
                blocks.append(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                )

        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ResNet/ResNeXt encoder wrapper extracting multi-scale features
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        encoder_map = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "resnext50_32x4d": models.resnext50_32x4d,
            "resnext101_32x8d": models.resnext101_32x8d,
            "resnext101_64x4d": models.resnext101_64x4d,
        }

        if encoder_name in encoder_map:
            weights = "IMAGENET1K_V1" if encoder_weights == "imagenet" else None
            backbone = encoder_map[encoder_name](weights=weights)
        elif encoder_name == "se_resnext50_32x4d":
            try:
                import timm

                pretrained = encoder_weights == "imagenet"
                backbone = timm.create_model(
                    "seresnext50_32x4d", pretrained=pretrained, features_only=False
                )
            except ImportError:
                raise ImportError(
                    "timm is required for SE-ResNeXt. Install with: pip install timm"
                )
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self._out_channels = self._get_output_channels(encoder_name)

    def _get_output_channels(self, encoder_name: str) -> List[int]:
        if encoder_name in ["resnet18", "resnet34"]:
            return [3, 64, 64, 128, 256, 512]
        elif encoder_name in [
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "resnext101_64x4d",
            "se_resnext50_32x4d",
        ]:
            return [3, 64, 256, 512, 1024, 2048]
        else:
            return [3, 64, 256, 512, 1024, 2048]

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        return features


# Main FPN model with ResNet encoder and feature pyramid
class FPN(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        activation: Optional[str] = None,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        dropout: float = 0.2,
        merge_policy: str = "add",
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.classes = classes
        self.activation = activation
        self.merge_policy = merge_policy

        self.encoder = ResNetEncoder(
            encoder_name=encoder_name, encoder_weights=encoder_weights
        )

        encoder_channels = self.encoder.out_channels

        self.p5 = nn.Conv2d(encoder_channels[5], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[4])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[3])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[2])

        self.s5 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=3
        )
        self.s4 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=2
        )
        self.s3 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=1
        )
        self.s2 = SegmentationBlock(
            pyramid_channels, segmentation_channels, n_upsamples=0
        )

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

        if merge_policy == "add":
            self.final_conv = nn.Conv2d(
                segmentation_channels, classes, kernel_size=1, padding=0
            )
        elif merge_policy == "cat":
            self.final_conv = nn.Conv2d(
                segmentation_channels * 4, classes, kernel_size=1, padding=0
            )
        else:
            raise ValueError(f"Unknown merge_policy: {merge_policy}")

        self.activation_fn = self._get_activation(activation)

    def _get_activation(self, activation: Optional[str]):
        if activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax2d":
            return nn.Softmax(dim=1)
        else:
            return nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        c2 = features[2]
        c3 = features[3]
        c4 = features[4]
        c5 = features[5]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        if self.merge_policy == "add":
            x = s5 + s4 + s3 + s2
        elif self.merge_policy == "cat":
            x = torch.cat([s5, s4, s3, s2], dim=1)

        x = self.dropout(x)
        x = self.final_conv(x)

        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        x = self.activation_fn(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.eval()

        with torch.no_grad():
            return self.forward(x)


if __name__ == "__main__":
    encoders = ["resnet50", "resnet18"]

    for encoder_name in encoders:
        print(f"\nTesting FPN with {encoder_name}...")

        model = FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
            classes=1,
            activation=None,
            pyramid_channels=256,
            segmentation_channels=128,
        )

        x = torch.randn(2, 3, 512, 512)
        output = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
