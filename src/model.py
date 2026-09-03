"""Change-detection U-Net architectures and imbalance-aware losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SiameseUNet(nn.Module):
    """U-Net supporting legacy early fusion and true Siamese feature fusion.

    ``early_fusion`` concatenates the RGB pairs into six input channels and is
    compatible with the existing checkpoint. ``siamese_diff`` runs both images
    through one shared encoder and decodes absolute feature differences.
    """

    def __init__(
        self,
        encoder_name="resnet34",
        pretrained=True,
        fusion_mode="early_fusion",
    ):
        super().__init__()
        if fusion_mode not in {"early_fusion", "siamese_diff"}:
            raise ValueError("fusion_mode must be 'early_fusion' or 'siamese_diff'")

        self.fusion_mode = fusion_mode
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=6 if fusion_mode == "early_fusion" else 3,
            classes=1,
            activation=None,
        )

    def forward(self, img_a, img_b):
        if self.fusion_mode == "early_fusion":
            return self.unet(torch.cat([img_a, img_b], dim=1))

        features_a = self.unet.encoder(img_a)
        features_b = self.unet.encoder(img_b)
        difference_features = [
            torch.abs(feature_a - feature_b)
            for feature_a, feature_b in zip(features_a, features_b)
        ]
        decoder_output = self.unet.decoder(difference_features)
        return self.unet.segmentation_head(decoder_output)


def _soft_dice_loss(logits, targets, smooth=1e-6):
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(2, 3))
    score = (2 * intersection + smooth) / (
        probabilities.sum(dim=(2, 3))
        + targets.sum(dim=(2, 3))
        + smooth
    )
    return 1 - score.mean()


class DiceBCELoss(nn.Module):
    """Stable baseline combining pixel-wise BCE with region-level Dice."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = _soft_dice_loss(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice


class FocalDiceLoss(nn.Module):
    """Focus learning on difficult pixels while preserving mask overlap."""

    def __init__(self, focal_weight=0.5, gamma=2.0, positive_alpha=0.75):
        super().__init__()
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.positive_alpha = positive_alpha

    def forward(self, logits, targets):
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        probability_of_target = (
            probabilities * targets + (1 - probabilities) * (1 - targets)
        )
        alpha = (
            self.positive_alpha * targets
            + (1 - self.positive_alpha) * (1 - targets)
        )
        focal = (
            alpha
            * (1 - probability_of_target).pow(self.gamma)
            * cross_entropy
        ).mean()
        dice = _soft_dice_loss(logits, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * dice


class TverskyLoss(nn.Module):
    """Overlap loss with explicit false-positive/false-negative weighting."""

    def __init__(self, false_positive_weight=0.4, false_negative_weight=0.6):
        super().__init__()
        self.fp_weight = false_positive_weight
        self.fn_weight = false_negative_weight

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        dims = (2, 3)
        true_positive = (probabilities * targets).sum(dim=dims)
        false_positive = (probabilities * (1 - targets)).sum(dim=dims)
        false_negative = ((1 - probabilities) * targets).sum(dim=dims)
        smooth = 1e-6
        score = (true_positive + smooth) / (
            true_positive
            + self.fp_weight * false_positive
            + self.fn_weight * false_negative
            + smooth
        )
        return 1 - score.mean()


def build_loss(name):
    """Construct a loss from a command-line friendly name."""
    if name == "dice_bce":
        return DiceBCELoss()
    if name == "focal_dice":
        return FocalDiceLoss()
    if name == "tversky":
        return TverskyLoss()
    raise ValueError(f"Unknown loss: {name}")
