"""Native-resolution tiled inference with optional flip augmentation."""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import SiameseUNet


_TRANSFORM = A.Compose(
    [
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ],
    additional_targets={"image2": "image"},
)


def _load_image(source):
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"))
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {source}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _detect_fusion_mode(state_dict):
    first_layer = state_dict.get("unet.encoder.conv1.weight")
    if first_layer is None:
        raise ValueError("Checkpoint does not contain the expected ResNet encoder")
    if first_layer.shape[1] == 6:
        return "early_fusion"
    if first_layer.shape[1] == 3:
        return "siamese_diff"
    raise ValueError(
        f"Unsupported checkpoint input channel count: {first_layer.shape[1]}"
    )


def _tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


class ChangeDetector:
    def __init__(self, checkpoint_path, device=None, encoder_name="resnet34"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        state_dict = _load_state_dict(checkpoint_path, self.device)
        self.fusion_mode = _detect_fusion_mode(state_dict)
        self.model = SiameseUNet(
            encoder_name=encoder_name,
            pretrained=False,
            fusion_mode=self.fusion_mode,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def _predict_tensor_batch(self, image_a, image_b, tta):
        if not tta:
            return torch.sigmoid(self.model(image_a, image_b))

        predictions = []
        for flip_h, flip_v in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            augmented_a = image_a
            augmented_b = image_b
            if flip_h:
                augmented_a = augmented_a.flip(-1)
                augmented_b = augmented_b.flip(-1)
            if flip_v:
                augmented_a = augmented_a.flip(-2)
                augmented_b = augmented_b.flip(-2)

            probability = torch.sigmoid(self.model(augmented_a, augmented_b))
            if flip_v:
                probability = probability.flip(-2)
            if flip_h:
                probability = probability.flip(-1)
            predictions.append(probability)
        return torch.stack(predictions).mean(dim=0)

    @torch.no_grad()
    def predict(
        self,
        before,
        after,
        threshold=0.5,
        img_size=256,
        tta=True,
        overlap=0.25,
        tile_batch_size=8,
    ):
        """Predict a full-resolution mask by blending overlapping tiles."""
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between 0 and 1")

        img_a = _load_image(before)
        img_b = _load_image(after)
        after_was_resized = img_a.shape != img_b.shape
        if after_was_resized:
            img_b = cv2.resize(
                img_b,
                (img_a.shape[1], img_a.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )

        original_height, original_width = img_a.shape[:2]
        pad_bottom = max(0, img_size - original_height)
        pad_right = max(0, img_size - original_width)
        if pad_bottom or pad_right:
            img_a_padded = cv2.copyMakeBorder(
                img_a, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101
            )
            img_b_padded = cv2.copyMakeBorder(
                img_b, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101
            )
        else:
            img_a_padded, img_b_padded = img_a, img_b

        height, width = img_a_padded.shape[:2]
        stride = max(1, int(round(img_size * (1 - overlap))))
        y_positions = _tile_positions(height, img_size, stride)
        x_positions = _tile_positions(width, img_size, stride)
        locations = [(y, x) for y in y_positions for x in x_positions]

        window_1d = np.hanning(img_size).astype(np.float32)
        blend_window = np.maximum(np.outer(window_1d, window_1d), 0.05)
        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        for start in range(0, len(locations), tile_batch_size):
            batch_locations = locations[start:start + tile_batch_size]
            tensors_a, tensors_b = [], []
            for top, left in batch_locations:
                tile_a = img_a_padded[top:top + img_size, left:left + img_size]
                tile_b = img_b_padded[top:top + img_size, left:left + img_size]
                transformed = _TRANSFORM(image=tile_a, image2=tile_b)
                tensors_a.append(transformed["image"])
                tensors_b.append(transformed["image2"])

            tensor_a = torch.stack(tensors_a).to(self.device)
            tensor_b = torch.stack(tensors_b).to(self.device)
            batch_probabilities = self._predict_tensor_batch(
                tensor_a, tensor_b, tta
            ).squeeze(1).cpu().numpy()

            for probability, (top, left) in zip(
                batch_probabilities, batch_locations
            ):
                region = np.s_[top:top + img_size, left:left + img_size]
                probability_sum[region] += probability * blend_window
                weight_sum[region] += blend_window

        probability_map = probability_sum / np.maximum(weight_sum, 1e-8)
        probability_map = probability_map[:original_height, :original_width]
        mask = (probability_map > threshold).astype(np.uint8)

        overlay = img_a.copy()
        overlay[mask == 1] = [255, 60, 60]
        heatmap = cv2.addWeighted(img_a, 0.55, overlay, 0.45, 0)

        return {
            "heatmap": heatmap,
            "mask": mask,
            "prob_map": probability_map,
            "change_pct": float(mask.mean() * 100),
            "after_was_resized": after_was_resized,
            "fusion_mode": self.fusion_mode,
            "tile_count": len(locations),
        }
