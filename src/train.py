"""
src/train.py
Full training loop with validation, checkpointing, and logging.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch_size 8 --lr 1e-4
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LEVIRDataset
from model   import SiameseUNet, DiceBCELoss
from metrics import iou_score, f1_score


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="data/raw",   type=str)
    p.add_argument("--save_dir",   default="models",     type=str)
    p.add_argument("--img_size",   default=256,          type=int)
    p.add_argument("--batch_size", default=8,            type=int)
    p.add_argument("--epochs",     default=40,           type=int)
    p.add_argument("--lr",         default=1e-4,         type=float)
    p.add_argument("--encoder",    default="resnet34",   type=str)
    p.add_argument("--num_workers",default=4,            type=int)
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Train / Val loop                                                            #
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for img_a, img_b, mask in tqdm(loader, desc="  train", leave=False):
        img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
        optimizer.zero_grad()
        logits = model(img_a, img_b)
        loss   = criterion(logits, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, total_f1 = 0.0, 0.0, 0.0
    for img_a, img_b, mask in tqdm(loader, desc="  val  ", leave=False):
        img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
        logits = model(img_a, img_b)
        loss   = criterion(logits, mask)
        total_loss += loss.item()
        total_iou  += iou_score(logits, mask)
        total_f1   += f1_score(logits, mask)
    n = len(loader)
    return total_loss / n, total_iou / n, total_f1 / n


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    train_ds = LEVIRDataset(args.data_dir, split="train", img_size=args.img_size)
    val_ds   = LEVIRDataset(args.data_dir, split="val",   img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ---- Model ---------------------------------------------------------------
    model     = SiameseUNet(encoder_name=args.encoder).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop -------------------------------------------------------
    best_iou = 0.0
    history  = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss              = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_iou": val_iou, "val_f1": val_f1,
        })

        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val   loss : {val_loss:.4f}  |  IoU : {val_iou:.4f}  |  F1 : {val_f1:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"  ✓ Saved best model (IoU={best_iou:.4f})")

    # Always save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))
    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")

    # Save training history
    import json
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
