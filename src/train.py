"""
src/train.py
Full training loop with validation, checkpointing, and logging.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch_size 8 --lr 1e-4
"""

import os
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LEVIRDataset
from model   import SiameseUNet, DiceBCELoss
from metrics import iou_score, f1_score


def load_existing_best(save_dir):
    """Return the best IoU already stored in save_dir, if one exists.

    New runs write a small metadata file next to the checkpoint. For checkpoints
    created by older versions of this script, history.json is used once to
    recover the score and bootstrap the metadata file.
    """
    checkpoint_path = os.path.join(save_dir, "best_model.pth")
    metadata_path = os.path.join(save_dir, "best_model_metadata.json")
    history_path = os.path.join(save_dir, "history.json")

    if not os.path.exists(checkpoint_path):
        return 0.0, None

    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return float(metadata["val_iou"]), "best_model_metadata.json"

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            previous_history = json.load(f)

        if previous_history:
            previous_best = max(previous_history, key=lambda row: row["val_iou"])
            metadata = {
                "epoch": int(previous_best["epoch"]),
                "val_iou": float(previous_best["val_iou"]),
                "val_f1": float(previous_best["val_f1"]),
                "val_loss": float(previous_best["val_loss"]),
                "source": "recovered from history.json",
            }
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return metadata["val_iou"], "history.json"

    # The checkpoint is valuable, but there is no trustworthy score with which
    # to compare it. Refuse to overwrite it until its metric is supplied.
    return float("inf"), "unscored checkpoint"


def save_best_metadata(save_dir, epoch, val_loss, val_iou, val_f1):
    """Persist the score associated with best_model.pth for future runs."""
    metadata = {
        "epoch": epoch,
        "val_iou": val_iou,
        "val_f1": val_f1,
        "val_loss": val_loss,
        "source": "training validation",
    }
    with open(os.path.join(save_dir, "best_model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #

def get_args():
    # Keep the CLI small and explicit so training runs are easy to reproduce.
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
        # Move the full batch to the target device before the forward pass.
        img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
        optimizer.zero_grad()
        # Forward, loss, backward, step: the standard supervised training cycle.
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
        # Validation mirrors training data flow but skips gradient tracking.
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
    # Resolve configuration first so every downstream component uses the same settings.
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the output directory up front so checkpoints and logs can be saved.
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Data ----------------------------------------------------------------
    # Train and validation splits are loaded independently to keep evaluation honest.
    train_ds = LEVIRDataset(args.data_dir, split="train", img_size=args.img_size)
    val_ds   = LEVIRDataset(args.data_dir, split="val",   img_size=args.img_size)

    # Shuffle training data, but keep validation deterministic.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # ---- Model ---------------------------------------------------------------
    # The model, loss, optimizer, and scheduler are initialized together so the
    # full optimization state is defined in one place.
    model     = SiameseUNet(encoder_name=args.encoder).to(device)
    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop -------------------------------------------------------
    # Start from the best score achieved across all earlier runs in this
    # directory. This prevents a weaker new run from overwriting the checkpoint.
    best_iou, best_source = load_existing_best(args.save_dir)
    if best_source == "unscored checkpoint":
        print(
            "WARNING: An existing best_model.pth has no saved validation score. "
            "It will be preserved and not overwritten."
        )
    elif best_source:
        print(f"Preserving existing global best IoU: {best_iou:.4f} ({best_source})")

    history  = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Run one full optimization pass, then evaluate on the held-out split.
        train_loss              = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Keep a compact per-epoch record for later analysis or plotting.
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
            save_best_metadata(
                args.save_dir,
                epoch=epoch,
                val_loss=val_loss,
                val_iou=val_iou,
                val_f1=val_f1,
            )
            print(f"  ✓ Saved new global best model (IoU={best_iou:.4f})")

    # Always save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, "last_model.pth"))
    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")

    # Save training history
    # Persist the epoch metrics as JSON so downstream scripts can inspect them
    # without needing to rerun training.
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
