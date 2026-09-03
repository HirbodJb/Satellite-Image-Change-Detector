"""Dataset loading and preprocessing for LEVIR-CD and LEVIR-CD+."""

import os

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def _list_triplets(split_dir):
    """Return matched (before, after, label) paths and reject broken pairs."""
    if not os.path.isdir(split_dir):
        return []

    triplets = []
    for filename in sorted(os.listdir(os.path.join(split_dir, "A"))):
        if not filename.lower().endswith(".png"):
            continue
        paths = (
            os.path.join(split_dir, "A", filename),
            os.path.join(split_dir, "B", filename),
            os.path.join(split_dir, "label", filename),
        )
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"Incomplete image pair for {filename}: missing {missing}"
            )
        triplets.append(paths)
    return triplets


def get_train_geometric_transforms():
    """Spatial augmentations shared by both dates and the change mask."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"image2": "image"},
    )


def get_photometric_transforms():
    """Mild date-specific appearance changes applied independently."""
    return A.Compose(
        [
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
                p=0.35,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.08,
                contrast_limit=0.08,
                p=0.20,
            ),
        ]
    )


def get_tensor_transform():
    """Apply ImageNet normalization and convert arrays to tensors."""
    return A.Compose(
        [
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        additional_targets={"image2": "image"},
    )


def get_val_transforms(img_size=256):
    """Deterministic center-crop pipeline used for model selection."""
    return A.Compose(
        [
            A.CenterCrop(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        additional_targets={"image2": "image"},
    )


class LEVIRDataset(Dataset):
    """Load aligned before/after image pairs and their binary change masks.

    LEVIR-CD+ has no validation directory. A deterministic portion of its
    training set is therefore reserved for validation; its official test set is
    used only when ``split="test"``.
    """

    def __init__(
        self,
        root_dir,
        split="train",
        img_size=256,
        positive_crop_probability=0.70,
        plus_val_fraction=0.10,
        split_seed=42,
        include_plus=True,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")
        if not 0.0 <= positive_crop_probability <= 1.0:
            raise ValueError("positive_crop_probability must be between 0 and 1")
        if not 0.0 <= plus_val_fraction < 1.0:
            raise ValueError("plus_val_fraction must be in [0, 1)")

        self.split = split
        self.img_size = img_size
        self.positive_crop_probability = positive_crop_probability

        filenames_primary = _list_triplets(os.path.join(root_dir, split))
        filenames_plus = []

        plus_root = os.path.join(root_dir, "levir_plus", "LEVIR-CD+")
        if include_plus and os.path.isdir(plus_root):
            if split == "test":
                filenames_plus = _list_triplets(os.path.join(plus_root, "test"))
            else:
                plus_train = _list_triplets(os.path.join(plus_root, "train"))
                rng = np.random.default_rng(split_seed)
                order = rng.permutation(len(plus_train))
                val_count = int(round(len(plus_train) * plus_val_fraction))
                val_indices = set(order[:val_count].tolist())
                if split == "val":
                    filenames_plus = [
                        item for index, item in enumerate(plus_train)
                        if index in val_indices
                    ]
                else:
                    filenames_plus = [
                        item for index, item in enumerate(plus_train)
                        if index not in val_indices
                    ]

        self.filenames = filenames_primary + filenames_plus
        self.geometric_transform = get_train_geometric_transforms()
        self.photometric_transform = get_photometric_transforms()
        self.tensor_transform = get_tensor_transform()
        self.val_transform = get_val_transforms(img_size)

        print(
            f"  [{split}] LEVIR-CD: {len(filenames_primary)} | "
            f"LEVIR-CD+: {len(filenames_plus)} | Total: {len(self.filenames)}"
        )

    def __len__(self):
        return len(self.filenames)

    def _change_aware_crop(self, img_a, img_b, mask):
        """Crop around change pixels most of the time, while retaining negatives."""
        crop = self.img_size
        height, width = mask.shape
        if height < crop or width < crop:
            pad_bottom = max(0, crop - height)
            pad_right = max(0, crop - width)
            border = (0, pad_bottom, 0, pad_right)
            img_a = cv2.copyMakeBorder(img_a, *border, cv2.BORDER_REFLECT_101)
            img_b = cv2.copyMakeBorder(img_b, *border, cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, *border, cv2.BORDER_CONSTANT, value=0)
            height, width = mask.shape

        changed_y, changed_x = np.where(mask > 0)
        use_positive = (
            changed_y.size > 0
            and np.random.random() < self.positive_crop_probability
        )

        if use_positive:
            chosen = np.random.randint(changed_y.size)
            pixel_y, pixel_x = int(changed_y[chosen]), int(changed_x[chosen])
            y_min = max(0, pixel_y - crop + 1)
            y_max = min(pixel_y, height - crop)
            x_min = max(0, pixel_x - crop + 1)
            x_max = min(pixel_x, width - crop)
            top = np.random.randint(y_min, y_max + 1)
            left = np.random.randint(x_min, x_max + 1)
        else:
            top = np.random.randint(0, height - crop + 1)
            left = np.random.randint(0, width - crop + 1)

        crop_slice = np.s_[top:top + crop, left:left + crop]
        return img_a[crop_slice], img_b[crop_slice], mask[crop_slice]

    def __getitem__(self, idx):
        path_a, path_b, path_label = self.filenames[idx]
        raw_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
        raw_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
        raw_mask = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE)
        if raw_a is None or raw_b is None or raw_mask is None:
            raise ValueError(f"Could not decode dataset sample: {self.filenames[idx]}")

        img_a = cv2.cvtColor(raw_a, cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(raw_b, cv2.COLOR_BGR2RGB)
        mask = (raw_mask > 128).astype(np.float32)

        if img_a.shape != img_b.shape or img_a.shape[:2] != mask.shape:
            raise ValueError(f"Unaligned sample dimensions: {self.filenames[idx]}")

        if self.split == "train":
            img_a, img_b, mask = self._change_aware_crop(img_a, img_b, mask)
            transformed = self.geometric_transform(
                image=img_a, image2=img_b, mask=mask
            )
            img_a = self.photometric_transform(image=transformed["image"])["image"]
            img_b = self.photometric_transform(image=transformed["image2"])["image"]
            transformed = self.tensor_transform(
                image=img_a, image2=img_b, mask=transformed["mask"]
            )
        else:
            transformed = self.val_transform(image=img_a, image2=img_b, mask=mask)

        return (
            transformed["image"],
            transformed["image2"],
            transformed["mask"].unsqueeze(0),
        )
