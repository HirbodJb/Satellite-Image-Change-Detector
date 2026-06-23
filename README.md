# 🛰️ Satellite Change Detector

A Siamese U-Net that detects surface changes between before/after satellite image pairs.
Built for the **LEVIR-CD** dataset. Deployed as a Streamlit web app.

---

## Project Structure

```
satellite-change-detector/
├── data/
│   └── raw/                    ← PUT LEVIR-CD DATASET HERE
│       ├── train/
│       │   ├── A/              ← before images (.png)
│       │   ├── B/              ← after  images (.png)
│       │   └── label/          ← binary masks  (.png)
│       ├── val/
│       │   ├── A/  B/  label/
│       └── test/
│           ├── A/  B/  label/
├── models/                     ← checkpoints saved here after training
├── src/
│   ├── dataset.py              ← Dataset + augmentations
│   ├── model.py                ← Siamese U-Net + DiceBCE loss
│   ├── metrics.py              ← IoU, F1, Precision, Recall
│   ├── train.py                ← Training loop
│   └── inference.py            ← Predictor class used by the app
├── app/
│   └── app.py                  ← Streamlit UI
├── notebooks/                  ← (optional) exploration notebooks
└── requirements.txt
```

---

## Step 1 — Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Python 3.9+ recommended. CUDA optional but speeds up training 10x.

---

## Step 2 — Download the Dataset

Go to: **https://justchenhao.github.io/LEVIR/**

1. Download **LEVIR-CD Dataset** (the 256×256 patch version is easiest)
2. Unzip and place files so the structure matches:

```
data/raw/
    train/A/    train/B/    train/label/
    val/A/      val/B/      val/label/
    test/A/     test/B/     test/label/
```

The images are RGB `.png` files. Labels are grayscale `.png` where **white = change, black = no change**.

> **Tip**: If LEVIR is unavailable, the **WHU-CD** dataset has identical structure.
> Download from: https://study.rsgis.whu.edu.cn/pages/download/

---

## Step 3 — Train the Model

```bash
cd satellite-change-detector

# Default training (40 epochs, ResNet-34, 256px, batch 8)
python src/train.py

# Custom options
python src/train.py \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --encoder resnet34 \
    --img_size 256

# If you have a GPU (highly recommended)
# PyTorch auto-detects CUDA — no flag needed
```

### Training output
- `models/best_model.pth` — saved whenever val IoU improves
- `models/last_model.pth` — always saved at the end
- `models/history.json`   — loss + metrics per epoch

### Expected results on LEVIR-CD
| Metric | Expected after 40 epochs |
|--------|--------------------------|
| IoU    | ~0.78 – 0.83             |
| F1     | ~0.87 – 0.91             |

Training time: ~25 min on a single GPU (T4/V100), ~3–4 hrs on CPU.

> **Free GPU options**: Google Colab (T4), Kaggle Notebooks (P100), vast.ai (~$0.20/hr)

---

## Step 4 — Run the App

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`

1. Upload a **Before** satellite image
2. Upload the matching **After** image
3. Click **DETECT CHANGES**
4. See the heatmap overlay, binary mask, probability map, and change %

---

## Step 5 — Deploy to Hugging Face Spaces (Free)

1. Create a free account at https://huggingface.co
2. Create a new **Space** → choose **Streamlit**
3. Upload all files + `models/best_model.pth`
4. Add a `README.md` with `sdk: streamlit` in the YAML header

Your app will be live at `https://huggingface.co/spaces/YOUR_NAME/satellite-change-detector`

---

## Architecture

```
Before image (3ch) ──┐
                      ├─► Concatenate (6ch) ──► Shared ResNet-34 Encoder
After  image (3ch) ──┘                                    │
                                                    U-Net Decoder
                                                          │
                                               1-channel sigmoid output
                                                 (change probability map)
```

**Loss**: 0.5 × BCEWithLogits + 0.5 × Dice  
**Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)  
**Scheduler**: Cosine Annealing  

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `--batch_size` to 4 or lower `--img_size` to 128 |
| `FileNotFoundError: data/raw/train/A` | Check your dataset folder structure matches the layout above |
| App shows "Model not found" | Train first, or set the correct path in the sidebar |
| Low IoU after training | Try `--lr 3e-4` and `--epochs 60`; also check mask values are 0/255 |

---
