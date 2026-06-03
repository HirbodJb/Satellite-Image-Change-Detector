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
    Expects the LEVIR-CD folder layout:
        data/raw/
            train/
                A/          <- before images  (*.png)
                B/          <- after  images  (*.png)
                label/      <- change masks   (*.png, binary 0/255)
            val/
                A/ B/ label/
            test/
                A/ B/ label/
    """

    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256):
        self.split_dir = os.path.join(root_dir, split)
        self.img_size  = img_size
        self.transform = get_train_transforms(img_size) if split == "train" \
                         else get_val_transforms(img_size)

        self.filenames = sorted(os.listdir(os.path.join(self.split_dir, "A")))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        img_a = cv2.cvtColor(
            cv2.imread(os.path.join(self.split_dir, "A", fname)), cv2.COLOR_BGR2RGB
        )
        img_b = cv2.cvtColor(
            cv2.imread(os.path.join(self.split_dir, "B", fname)), cv2.COLOR_BGR2RGB
        )
        mask = cv2.imread(
            os.path.join(self.split_dir, "label", fname), cv2.IMREAD_GRAYSCALE
        )

        # Binarize mask (LEVIR uses 255 for change, 0 for no-change)
        mask = (mask > 128).astype(np.float32)

        transformed = self.transform(image=img_a, image2=img_b, mask=mask)
        img_a_t = transformed["image"]
        img_b_t = transformed["image2"]
        mask_t  = transformed["mask"].unsqueeze(0)   # (1, H, W)

        return img_a_t, img_b_t, mask_t