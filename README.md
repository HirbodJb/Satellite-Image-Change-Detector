# 🛰️ Satellite Change Detector

> Deep learning model that detects and quantifies surface changes between before/after satellite image pairs — deployed as a live interactive web app.

**[🚀 Live Demo](https://huggingface.co/spaces/HirbodJB/satellite-change-detector)** · **[💻 GitHub](https://github.com/HirbodJB/satellite-change-detector)**

---

![Demo Results](assets/demo_result.png)
*Left: Heatmap overlay highlighting detected changes in red. Center: Binary change mask. Right: Per-pixel change probability map.*

---

## What It Does

Upload two satellite images of the same location taken at different times. The model analyzes every pixel and produces:

- **Heatmap overlay** — red highlights painted directly on the before image
- **Binary change mask** — precise pixel-level map of what changed
- **Change probability map** — per-pixel model confidence from 0.0 to 1.0
- **Change percentage** — quantified area changed as a metric

Real-world applications include monitoring deforestation, tracking urban expansion, detecting flood or disaster damage, and measuring construction growth over time.

---

## Demo

| Before | After | Detected Changes |
|--------|-------|-----------------|
| ![before](assets/before.png) | ![after](assets/after.png) | ![result](assets/heatmap.png) |

*Example: Entire residential neighborhood constructed between image captures. Model correctly identifies new building footprints across the scene.*

---

## Results

Trained on **[LEVIR-CD](https://justchenhao.github.io/LEVIR/)** and **[LEVIR-CD+](https://github.com/S2Looking/Dataset)** high-resolution (1024×1024, 0.5m/pixel) Google Earth image pairs. A deterministic 10% of LEVIR-CD+ training data is held out for validation, and its official test split is never used for model selection.

| Metric | Score |
|--------|-------|
| **Current baseline validation IoU** | **0.7829** |
| **Current baseline validation F1** | **0.8783** |
| **Training Pairs** | 1,018 (445 LEVIR-CD + 573 LEVIR-CD+) |
| **Validation Pairs** | 128 (64 LEVIR-CD + 64 held-out LEVIR-CD+) |

---

## Architecture

```
Before image (3ch) ──► Shared ResNet-34 encoder ──► feature maps ──┐
                                                                  ├─► absolute differences
After  image (3ch) ──► Shared ResNet-34 encoder ──► feature maps ──┘          │
                                                                               ▼
                                                                    U-Net decoder
                                                                               │
                                                                    1-channel logits
```

**Design decisions:**
- **True Siamese design** — both dates pass separately through one shared encoder, and the decoder receives their absolute feature differences
- **Pretrained ResNet-34 encoder** — transfer learning from 1.2M ImageNet photos means strong visual understanding from epoch 1
- **U-Net decoder with skip connections** — preserves fine spatial detail that gets lost during encoding, critical for pixel-precise masks
- **Focal + Dice loss** — focal loss emphasizes difficult pixels while Dice handles sparse change regions
- **Change-aware crops** — 70% of eligible training crops are guaranteed to contain real change pixels while random negatives are retained
- **Test Time Augmentation (TTA)** — at inference, predictions are averaged across 4 flipped versions of each image pair for more robust results
- **Tiled inference** — overlapping native-resolution tiles are blended instead of shrinking the entire uploaded scene

**Training config:**
- Encoder: ResNet-34 (ImageNet pretrained)
- Fusion: shared-encoder absolute feature difference (`siamese_diff`)
- Loss: Focal + Dice (`focal_dice`)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: Cosine Annealing
- Epochs: 100 · Batch size: 8 · Image size: 256×256
- Hardware: RTX 5070 Ti (~25 min)

---

## Run Locally

**1. Clone and install:**
```bash
git clone https://github.com/HirbodJB/satellite-change-detector
cd satellite-change-detector

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

**2. Download the dataset:**

Go to [LEVIR-CD](https://justchenhao.github.io/LEVIR/) → Google Drive → download `train.zip`, `val.zip`, `test.zip` and unzip into:

```
data/raw/
    train/A/    train/B/    train/label/
    val/A/      val/B/      val/label/
    test/A/     test/B/     test/label/
```

Optionally download [LEVIR-CD+](https://drive.google.com/file/d/1JamSsxiytXdzAIk6VDVWfc-OsX-81U81) and place at:
```
data/raw/levir_plus/LEVIR-CD+/
    train/A/    train/B/    train/label/
    test/A/     test/B/     test/label/
```

**3. Train:**
```bash
python src/train.py --epochs 100 --lr 1e-4 --img_size 256 --batch_size 8 --encoder resnet34 --fusion_mode siamese_diff --loss focal_dice
```

**4. Run the app:**
```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501` — upload a before and after image, hit Detect Changes.

---

## Project Structure

```
satellite-change-detector/
├── src/
│   ├── dataset.py       ← LEVIR-CD + LEVIR-CD+ dataloader + augmentations
│   ├── model.py         ← Early-fusion/Siamese U-Nets + selectable losses
│   ├── metrics.py       ← Globally aggregated IoU, F1, Precision, Recall
│   ├── train.py         ← Reproducible training + cross-run best checkpointing
│   └── inference.py     ← Native-resolution tiled prediction + TTA
├── app/
│   └── app.py           ← Streamlit UI
├── assets/              ← README screenshots
├── data/raw/            ← Dataset goes here (not tracked by git)
├── models/              ← Saved checkpoints (not tracked by git)
└── requirements.txt
```

---

## Stack

`PyTorch` · `segmentation-models-pytorch` · `OpenCV` · `Albumentations` · `Streamlit` · `Hugging Face Spaces`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `--batch_size` to 4 |
| `ModuleNotFoundError` | Activate venv and run `pip install -r requirements.txt` |
| App shows "Model not found" | Check sidebar model path matches where `best_model.pth` is saved |
| Low IoU after training | Try `--lr 3e-4` and `--epochs 80`; verify mask pixel values are 0/255 |

---

## License

MIT — built by [Hirbod Jabbarnezhad](https://github.com/HirbodJB)
