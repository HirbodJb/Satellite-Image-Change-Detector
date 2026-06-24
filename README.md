# 🛰️ Satellite Change Detector

> Deep learning model that detects and quantifies surface changes between before/after satellite image pairs — deployed as a live interactive web app.

**[🚀 Live Demo](https://huggingface.co/spaces/HirbodJB/satellite-change-detector)** · **[📦 Model Weights](https://huggingface.co/spaces/HirbodJB/satellite-change-detector/tree/main)**

---

![Demo Results](assets/demo_result.png)
*Left: Heatmap overlay highlighting detected changes in red. Center: Binary change mask. Right: Per-pixel change probability map.*

---

## What It Does

Upload two satellite images of the same location taken at different times. The model analyzes every pixel and produces:

- **Heatmap overlay** — red highlights on areas that changed
- **Binary change mask** — precise pixel-level change map
- **Change probability map** — model confidence per pixel (0.0–1.0)
- **Change percentage** — quantified area changed

Real-world use cases include monitoring deforestation, tracking urban expansion, detecting flood or disaster damage, and measuring construction growth.

---

## Demo

| Before | After | Detected Changes |
|--------|-------|-----------------|
| ![before](assets/before.png) | ![after](assets/after.png) | ![result](assets/heatmap.png) |

*Example: Entire residential neighborhood constructed between image captures. Model correctly identifies new building footprints across the scene.*

---

## Architecture

```
Before image (3ch) ──┐
                      ├──► Concatenate (6ch) ──► ResNet-34 Encoder (pretrained ImageNet)
After  image (3ch) ──┘              │                        │
                                    │              5 feature scales extracted
                                    │                        │
                                    └──────► U-Net Decoder (skip connections)
                                                             │
                                              1-channel sigmoid output
                                              (per-pixel change probability)
```

**Why this architecture?**
- **Siamese design** — both images processed with shared weights, so the model learns to compare rather than memorize
- **Pretrained ResNet-34 encoder** — transfer learning from 1.2M ImageNet images means the model understands visual structure from day one
- **U-Net decoder** — skip connections preserve fine spatial detail lost during encoding, critical for pixel-precise masks

---

## Training Results

Trained on **[LEVIR-CD](https://justchenhao.github.io/LEVIR/)** — 637 high-resolution (1024×1024, 0.5m/pixel) Google Earth image pairs spanning 5–14 years of urban change.

| Metric | Score |
|--------|-------|
| **IoU (Jaccard)** | 0.68 |
| **F1 Score** | 0.79 |
| **Val Loss** | 0.37 |

**Training config:**
- Encoder: ResNet-34 (ImageNet pretrained)
- Loss: 0.5 × BCE + 0.5 × Dice
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: Cosine Annealing
- Epochs: 100 · Batch size: 8 · Image size: 256×256
- Hardware: RTX 5070 Ti (~20 min training)

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

Go to [LEVIR-CD](https://justchenhao.github.io/LEVIR/) → Google Drive → download `train.zip`, `val.zip`, `test.zip`

Unzip into:
```
data/raw/
    train/A/    train/B/    train/label/
    val/A/      val/B/      val/label/
    test/A/     test/B/     test/label/
```

**3. Train:**
```bash
python src/train.py --epochs 100 --lr 1e-4 --img_size 256 --batch_size 8 --encoder resnet34
```
Saves `models/best_model.pth` whenever validation IoU improves.

**4. Run the app:**
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501`

---

## Project Structure

```
satellite-change-detector/
├── src/
│   ├── dataset.py       ← LEVIR-CD dataloader + augmentations
│   ├── model.py         ← Siamese U-Net + DiceBCE loss
│   ├── metrics.py       ← IoU, F1, Precision, Recall
│   ├── train.py         ← Training loop with checkpointing
│   └── inference.py     ← Predictor class used by the app
├── app/
│   └── app.py           ← Streamlit UI
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
| `ModuleNotFoundError` | Make sure venv is activated and `pip install -r requirements.txt` was run |
| App shows "Model not found" | Check model path in sidebar matches where `best_model.pth` is saved |
| Low IoU | Try `--lr 3e-4` and `--epochs 80`; verify mask pixel values are 0/255 |

---

## License

MIT
