"""
src/model.py
Siamese U-Net for binary change detection.
Both branches share the same ResNet-34 encoder (weights tied).
The decoder receives the absolute difference of the two encoder feature maps.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseUNet(nn.Module):
    """
    Architecture:
        - Shared encoder  : ResNet-34 pretrained on ImageNet
        - Feature fusion  : |f_A - f_B|  at every decoder level
        - Decoder + head  : standard U-Net decoder → 1-channel sigmoid output
    """

    def __init__(self, encoder_name: str = "resnet34", pretrained: bool = True):
        super().__init__()

        weights = "imagenet" if pretrained else None

        # We build ONE U-Net; the encoder is used twice (shared weights)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=6,          # concatenate both images channel-wise → 3+3=6
            classes=1,
            activation=None,        # raw logits; BCEWithLogitsLoss handles sigmoid
        )

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_a: (B, 3, H, W) before image
            img_b: (B, 3, H, W) after  image
        Returns:
            logits: (B, 1, H, W)
        """
        x = torch.cat([img_a, img_b], dim=1)   # (B, 6, H, W)
        return self.unet(x)


# --------------------------------------------------------------------------- #
#  Loss                                                                        #
# --------------------------------------------------------------------------- #

class DiceBCELoss(nn.Module):
    """Weighted combination of BCE + Dice loss — robust for imbalanced masks."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss  = self.bce(logits, targets)

        probs     = torch.sigmoid(logits)
        smooth    = 1e-6
        inter     = (probs * targets).sum(dim=(2, 3))
        dice_loss = 1 - (2 * inter + smooth) / (
            probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
        )
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
