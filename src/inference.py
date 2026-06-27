"""
Inference helpers for the satellite change detector.

This module loads a trained Siamese U-Net checkpoint, runs prediction on a
before/after image pair, and returns both a visual overlay and a simple change
percentage summary.
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import SiameseUNet


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
# The inference path is fully deterministic. No augmentation is applied here;
# the inputs are normalized and converted to tensors exactly once.

_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={"image2": "image"})


def _load_and_resize(path: str, size: int = 256) -> np.ndarray:
    # OpenCV loads images in BGR, so convert to RGB before resizing.
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


def _pil_to_array(pil_img: Image.Image, size: int = 256) -> np.ndarray:
    # Mirror the file-based loader for PIL images so both input types follow
    # the same resize and color conversion behavior.
    img = pil_img.convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
# Thin wrapper around the trained model that prepares inputs, runs inference,
# applies thresholding, and builds the visualization outputs.

class ChangeDetector:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        # Keep model loading simple and explicit: instantiate the network,
        # load the checkpoint, move it to the requested device, and switch to
        # evaluation mode.
        self.device = torch.device(device)
        self.model  = SiameseUNet()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        before,
        after,
        threshold: float = 0.5,
        img_size: int    = 256,
        tta: bool        = True,
    ):
        # Support either PIL images or file paths, then route both through the
        # matching loader so the downstream tensor pipeline stays identical.
        load = _pil_to_array if isinstance(before, Image.Image) else _load_and_resize

        img_a = load(before,  img_size) if isinstance(before, Image.Image) else _load_and_resize(before,  img_size)
        img_b = load(after,   img_size) if isinstance(after,  Image.Image) else _load_and_resize(after,   img_size)

        # Albumentations handles normalization and tensor conversion for both
        # images at once so the pair remains perfectly aligned.
        t = _TRANSFORM(image=img_a, image2=img_b)
        a_t = t["image"].unsqueeze(0).to(self.device)
        b_t = t["image2"].unsqueeze(0).to(self.device)

        if tta:
            # Run 4 augmented versions and average predictions
            # Test-time augmentation: evaluate the original pair plus flipped
            # variants, then average the probability maps after undoing flips.
            preds = []
            for flip_h, flip_v in [(False,False),(True,False),(False,True),(True,True)]:
                a_aug = a_t.clone()
                b_aug = b_t.clone()
                if flip_h:
                    a_aug = a_aug.flip(-1)
                    b_aug = b_aug.flip(-1)
                if flip_v:
                    a_aug = a_aug.flip(-2)
                    b_aug = b_aug.flip(-2)
                logits = self.model(a_aug, b_aug)
                prob   = torch.sigmoid(logits)
                if flip_h:
                    prob = prob.flip(-1)
                if flip_v:
                    prob = prob.flip(-2)
                preds.append(prob)
            prob_map = torch.stack(preds).mean(0).squeeze().cpu().numpy()
        else:
            # Fast path: run a single forward pass with no test-time
            # augmentation.
            logits   = self.model(a_t, b_t)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Convert probabilities into a binary change mask using the caller's
        # threshold.
        mask = (prob_map > threshold).astype(np.uint8)

        # Build a simple red overlay on the original input for quick visual
        # inspection of the predicted change areas.
        bg      = img_a.copy()
        overlay = bg.copy()
        overlay[mask == 1] = [255, 60, 60]
        alpha   = 0.45
        heatmap = cv2.addWeighted(bg, 1 - alpha, overlay, alpha, 0)

        # Report the fraction of pixels classified as change, as a percentage.
        change_pct = float(mask.sum()) / mask.size * 100

        return {
            "heatmap"    : heatmap,
            "mask"       : mask,
            "prob_map"   : prob_map,
            "change_pct" : change_pct,
        }