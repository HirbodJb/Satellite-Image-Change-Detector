"""
src/inference.py
Load a trained model and run prediction on two image files.
Returns a heatmap overlay and a change-percentage metric.
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import SiameseUNet


# --------------------------------------------------------------------------- #
#  Preprocessing (no augmentation — deterministic)                            #
# --------------------------------------------------------------------------- #

_TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={"image2": "image"})


def _load_and_resize(path: str, size: int = 256) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


def _pil_to_array(pil_img: Image.Image, size: int = 256) -> np.ndarray:
    img = pil_img.convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


# --------------------------------------------------------------------------- #
#  Predictor                                                                   #
# --------------------------------------------------------------------------- #

class ChangeDetector:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model  = SiameseUNet()
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        before,          # str path OR PIL.Image
        after,           # str path OR PIL.Image
        threshold: float = 0.5,
        img_size: int    = 256,
    ):
        """
        Returns
        -------
        result_dict : dict with keys
            'heatmap'      : np.ndarray (H, W, 3)  — colour overlay on 'before'
            'mask'         : np.ndarray (H, W)     — binary 0/1
            'prob_map'     : np.ndarray (H, W)     — float 0–1
            'change_pct'   : float                 — % pixels changed
        """
        load = _pil_to_array if isinstance(before, Image.Image) else _load_and_resize

        img_a = load(before,  img_size) if isinstance(before, Image.Image) else _load_and_resize(before,  img_size)
        img_b = load(after,   img_size) if isinstance(after,  Image.Image) else _load_and_resize(after,   img_size)

        t = _TRANSFORM(image=img_a, image2=img_b)
        a_t = t["image"].unsqueeze(0).to(self.device)
        b_t = t["image2"].unsqueeze(0).to(self.device)

        logits   = self.model(a_t, b_t)                        # (1,1,H,W)
        prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H,W)
        mask     = (prob_map > threshold).astype(np.uint8)

        # ---- Build colour overlay -------------------------------------------
        # Use original before image as background
        bg     = img_a.copy()
        overlay = bg.copy()

        # Changed pixels → bright red
        overlay[mask == 1] = [255, 60, 60]

        # Blend
        alpha   = 0.45
        heatmap = cv2.addWeighted(bg, 1 - alpha, overlay, alpha, 0)

        change_pct = float(mask.sum()) / mask.size * 100

        return {
            "heatmap"    : heatmap,
            "mask"       : mask,
            "prob_map"   : prob_map,
            "change_pct" : change_pct,
        }