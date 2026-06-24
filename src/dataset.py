"""
src/dataset.py
Handles loading and preprocessing of LEVIR-CD dataset image pairs.
"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# --------------------------------------------------------------------------- #
#  Augmentation pipelines                                                      #
# --------------------------------------------------------------------------- #

def get_train_transforms(img_size=256):
    return A.Compose([
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"image2": "image"})


def get_val_transforms(img_size=256):
    return A.Compose([
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={"image2": "image"})


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class LEVIRDataset(Dataset):
    """
    Loads LEVIR-CD and optionally LEVIR-CD+ combined.
    
    LEVIR-CD layout:   data/raw/train/A, B, label
    LEVIR-CD+ layout:  data/raw/levir_plus/train/A, B, label
    """

    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256):
        self.img_size  = img_size
        self.transform = get_train_transforms(img_size) if split == "train" \
                         else get_val_transforms(img_size)

        # Primary dataset (LEVIR-CD)
        primary_dir = os.path.join(root_dir, split)
        filenames_primary = [
            (os.path.join(primary_dir, "A", f),
             os.path.join(primary_dir, "B", f),
             os.path.join(primary_dir, "label", f))
            for f in sorted(os.listdir(os.path.join(primary_dir, "A")))
        ]

        # LEVIR-CD+ (only train and test splits exist)
        plus_split  = "train" if split == "train" else "test"
        plus_dir    = os.path.join(root_dir, "..", "levir_plus", plus_split)
        filenames_plus = []
        if os.path.exists(plus_dir):
            filenames_plus = [
                (os.path.join(plus_dir, "A", f),
                 os.path.join(plus_dir, "B", f),
                 os.path.join(plus_dir, "label", f))
                for f in sorted(os.listdir(os.path.join(plus_dir, "A")))
            ]

        self.filenames = filenames_primary + filenames_plus
        print(f"  [{split}] LEVIR-CD: {len(filenames_primary)} | LEVIR-CD+: {len(filenames_plus)} | Total: {len(self.filenames)}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path_a, path_b, path_label = self.filenames[idx]

        img_a = cv2.cvtColor(cv2.imread(path_a), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(path_b), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 128).astype(np.float32)

        transformed = self.transform(image=img_a, image2=img_b, mask=mask)
        img_a_t = transformed["image"]
        img_b_t = transformed["image2"]
        mask_t  = transformed["mask"].unsqueeze(0)

        return img_a_t, img_b_t, mask_t