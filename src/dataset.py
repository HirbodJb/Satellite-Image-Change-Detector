"""
src/dataset.py
==============
Dataset loading and preprocessing pipeline for satellite change detection.

Supports two datasets with identical folder structures:
  - LEVIR-CD  : 637 image pairs, primary benchmark dataset
  - LEVIR-CD+ : 985 image pairs, extended version with more regions and years

Both datasets contain 1024×1024 RGB Google Earth image pairs at 0.5m/pixel
resolution, with binary grayscale labels where white (255) = changed and
black (0) = no change.

Expected directory layout:
    data/raw/
        train/  A/  B/  label/      ← LEVIR-CD training split
        val/    A/  B/  label/      ← LEVIR-CD validation split
        test/   A/  B/  label/      ← LEVIR-CD test split
        levir_plus/
            LEVIR-CD+/
                train/  A/  B/  label/   ← LEVIR-CD+ training split
                test/   A/  B/  label/   ← LEVIR-CD+ test split

Note: LEVIR-CD+ has no val split, so it is merged with train during training
and with test during evaluation.
"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Augmentation pipelines
#
# Both pipelines use `additional_targets={"image2": "image"}` so that the
# before (image) and after (image2) photos receive the SAME spatial transform
# — keeping them geometrically aligned. Color jitter is applied independently
# to simulate sensor/lighting variation between capture dates.
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 256) -> A.Compose:
    """
    Augmentation pipeline used during training.

    Applies spatial transforms (crops, flips, rotations) identically to both
    images and the mask to preserve alignment, plus mild color jitter on the
    images only to simulate real-world lighting and seasonal variation.

    Args:
        img_size: Output crop size in pixels (default 256 to match LEVIR-CD).

    Returns:
        An Albumentations Compose pipeline.
    """
    return A.Compose([
        # Random crop extracts a 256×256 patch from the 1024×1024 source image,
        # effectively multiplying dataset size by up to 16×
        A.RandomCrop(img_size, img_size),

        # Geometric flips — applied identically to image, image2, and mask
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Color jitter on images only (mask is unaffected by Albumentations design)
        # Simulates seasonal color shifts, different sensors, and lighting angles
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),

        # Normalize to ImageNet mean/std — required because the ResNet-34 encoder
        # was pretrained on ImageNet with these exact statistics
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        # Convert HWC numpy arrays to CHW PyTorch tensors
        ToTensorV2(),

    ], additional_targets={"image2": "image"})  # apply same spatial ops to both images


def get_val_transforms(img_size: int = 256) -> A.Compose:
    """
    Deterministic pipeline used during validation and testing.

    Uses CenterCrop instead of RandomCrop so results are reproducible
    across evaluation runs. No color jitter or geometric randomness.

    Args:
        img_size: Output crop size in pixels (default 256).

    Returns:
        An Albumentations Compose pipeline.
    """
    return A.Compose([
        # Center crop is deterministic — ensures consistent evaluation
        A.CenterCrop(img_size, img_size),

        # Same ImageNet normalization as training
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ToTensorV2(),

    ], additional_targets={"image2": "image"})


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LEVIRDataset(Dataset):
    """
    PyTorch Dataset that loads LEVIR-CD and optionally LEVIR-CD+ image pairs.

    Each sample consists of:
        img_a  : before image tensor  (3, H, W) — normalized float32
        img_b  : after  image tensor  (3, H, W) — normalized float32
        mask   : change mask tensor   (1, H, W) — binary float32 (0.0 or 1.0)

    The two datasets are concatenated at the filename level before any loading
    occurs, so DataLoader workers see a single flat list of (A, B, label) paths.

    If the LEVIR-CD+ directory does not exist, the dataset silently falls back
    to LEVIR-CD only — no code changes required.
    """

    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256):
        """
        Args:
            root_dir : Path to the data/raw/ directory.
            split    : One of "train", "val", or "test".
            img_size : Spatial resolution passed to the transform pipelines.
        """
        self.img_size = img_size

        # Select augmentation pipeline based on split
        # Val and test use the deterministic center-crop pipeline
        self.transform = (
            get_train_transforms(img_size) if split == "train"
            else get_val_transforms(img_size)
        )

        # ── Primary dataset: LEVIR-CD ────────────────────────────────────────
        primary_dir = os.path.join(root_dir, split)
        filenames_primary = [
            (
                os.path.join(primary_dir, "A", f),      # before image path
                os.path.join(primary_dir, "B", f),      # after  image path
                os.path.join(primary_dir, "label", f),  # change mask path
            )
            for f in sorted(os.listdir(os.path.join(primary_dir, "A")))
            if f.endswith('.png')  # filter out .DS_Store and other non-image files
        ]

        # ── Secondary dataset: LEVIR-CD+ ────────────────────────────────────
        # LEVIR-CD+ only has "train" and "test" splits (no "val"),
        # so we map val → test to avoid an empty directory error.
        plus_split = "train" if split == "train" else "test"
        plus_dir   = os.path.join(root_dir, "levir_plus", "LEVIR-CD+", plus_split)

        filenames_plus = []
        if os.path.exists(plus_dir):
            # Only loaded if the LEVIR-CD+ directory is present on disk
            filenames_plus = [
                (
                    os.path.join(plus_dir, "A", f),
                    os.path.join(plus_dir, "B", f),
                    os.path.join(plus_dir, "label", f),
                )
                for f in sorted(os.listdir(os.path.join(plus_dir, "A")))
                if f.endswith('.png')
            ]

        # Combine both datasets into a single flat list of file path tuples
        self.filenames = filenames_primary + filenames_plus

        print(
            f"  [{split}] LEVIR-CD: {len(filenames_primary)} | "
            f"LEVIR-CD+: {len(filenames_plus)} | "
            f"Total: {len(self.filenames)}"
        )

    def __len__(self) -> int:
        """Returns total number of image pairs across both datasets."""
        return len(self.filenames)

    def __getitem__(self, idx: int):
        """
        Load, decode, and transform one image pair.

        Args:
            idx: Index into the combined filename list.

        Returns:
            img_a_t : Before image tensor (3, H, W)
            img_b_t : After  image tensor (3, H, W)
            mask_t  : Change mask tensor  (1, H, W) — values 0.0 or 1.0
        """
        path_a, path_b, path_label = self.filenames[idx]

        # Load images as RGB (OpenCV reads BGR by default, so we convert)
        img_a = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2RGB)

        # Load mask as single-channel grayscale
        mask = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE)

        # Binarize: LEVIR labels use 255 for change, 0 for no-change.
        # Convert to float32 so BCEWithLogitsLoss can consume it directly.
        mask = (mask > 128).astype(np.float32)

        # Apply the same spatial transform to both images and the mask.
        # Color transforms only affect the images, not the mask.
        transformed = self.transform(image=img_a, image2=img_b, mask=mask)
        img_a_t = transformed["image"]   # (3, H, W) float32 tensor
        img_b_t = transformed["image2"]  # (3, H, W) float32 tensor
        mask_t  = transformed["mask"].unsqueeze(0)  # (1, H, W) — add channel dim

        return img_a_t, img_b_t, mask_t
