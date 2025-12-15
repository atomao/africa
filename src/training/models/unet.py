# U-Net with ResNet/ResNeXt encoders - compatible with segmentation_models_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import torchvision.models as models


# Decoder block with upsampling and skip connections
class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=not use_batchnorm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


# Final segmentation head with optional activation
class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[str] = None,
        upsampling: int = 1,
    ):
        super().__init__()

        self.upsampling = upsampling
        self.activation = activation

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        if self.upsampling > 1:
            x = F.interpolate(
                x, scale_factor=self.upsampling, mode="bilinear", align_corners=False
            )

        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "softmax2d":
            x = F.softmax(x, dim=1)

        return x


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


# Main U-Net model with ResNet encoder and decoder path
class Unet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        classes: int = 1,
        activation: Optional[str] = None,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_use_batchnorm: bool = True,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.classes = classes
        self.activation = activation

        self.encoder = ResNetEncoder(
            encoder_name=encoder_name, encoder_weights=encoder_weights
        )

        encoder_channels = self.encoder.out_channels

        self.decoder_blocks = nn.ModuleList()

        in_channels = encoder_channels[5]
        skip_channels = encoder_channels[4]
        out_channels = decoder_channels[0]

        self.decoder_blocks.append(
            DecoderBlock(
                in_channels=in_channels,
                skip_channels=skip_channels,
                out_channels=out_channels,
                use_batchnorm=decoder_use_batchnorm,
            )
        )

        for i in range(1, len(decoder_channels)):
            in_channels = decoder_channels[i - 1]
            skip_channels = encoder_channels[4 - i]
            out_channels = decoder_channels[i]

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    use_batchnorm=decoder_use_batchnorm,
                )
            )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        x = features[5]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = 4 - i
            skip = features[skip_idx] if skip_idx > 0 else None
            x = decoder_block(x, skip)

        x = self.segmentation_head(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.eval()

        with torch.no_grad():
            return self.forward(x)


UNet = Unet


if __name__ == "__main__":
    encoders = ["resnet50", "resnet18"]

    for encoder_name in encoders:
        print(f"\nTesting {encoder_name}...")

        model = Unet(
            encoder_name=encoder_name, encoder_weights=None, classes=1, activation=None
        )

        x = torch.randn(2, 3, 512, 512)
        output = model(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
